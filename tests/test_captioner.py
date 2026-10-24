"""How Auto-Caption and Enhance prompt read an LLM server's replies: thinking left out, a reply cut off at
max_tokens asked for again with more room, and empty or still-cut-off replies reported, never saved. And what a
caption style sends: its system prompt before the image, and its own temperature and max tokens.

    .venv/bin/python -m pytest tests/test_captioner.py -v
"""

import json
import subprocess

import httpx
import pytest
from PIL import Image

from dainamo.server import captioner

LLM = {"endpoint": "http://llm/v1", "model": "", "api_key": "", "max_tokens": 300, "temperature": 0.2,
       "prompt": "Describe this image."}


def server(*replies):
    """A client whose server gives ``replies`` in turn (a dict is a whole response, an int an error status, a
    string a finished reply), and the request bodies it got."""
    got, queue = [], list(replies)

    def handle(request):
        got.append(json.loads(request.content))
        reply = queue.pop(0)
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": {"code": reply, "message": "the request exceeds the "
                                                         "available context size, try increasing it"}})
        if isinstance(reply, str):
            reply = {"choices": [{"message": {"content": reply}, "finish_reason": "stop"}]}
        return httpx.Response(200, json=reply)

    return httpx.Client(transport=httpx.MockTransport(handle)), got


def cut_off(content, used, prompt=500, **message):
    return {"choices": [{"message": {"content": content, **message}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": used}}


def ask(client):
    return captioner._ask(LLM, client, LLM["prompt"], [Image.new("RGB", (32, 32))])


def test_thinking_is_left_out_of_the_reply():
    for reply in ("<think>a cat, probably</think>\n\nA cat on a mat.",
                  "The user wants a caption.\n</think>\n\nA cat on a mat.",  # the template opened the block
                  {"choices": [{"message": {"content": "A cat on a mat.", "reasoning_content": "a cat..."},
                                "finish_reason": "stop"}]}):
        client, _ = server(reply)
        assert ask(client) == "A cat on a mat."


def test_a_cut_off_reply_is_asked_for_again_with_more_room():
    client, got = server(cut_off("A cat sits on a woven mat in the warm light of", 300), "A cat on a mat, in full.")
    assert ask(client) == "A cat on a mat, in full."
    assert [g["max_tokens"] for g in got] == [300, 300 * captioner.RETRY_ROOM]


def test_empty_or_still_cut_off_replies_are_errors():
    client, _ = server("")
    with pytest.raises(RuntimeError, match="no text"):
        ask(client)
    client, _ = server("<think>still thinking when it stopped")
    with pytest.raises(RuntimeError, match="no text"):
        ask(client)
    client, got = server(cut_off("A cat sits", 300), cut_off("A cat sits on", 1200))
    with pytest.raises(RuntimeError, match="cut off at 1200 tokens: raise Max tokens"):
        ask(client)
    # a reasoning model that spent all of it thinking
    client, _ = server(cut_off("", 300, reasoning_content="hmm"), cut_off("", 1200, reasoning_content="hmm..."))
    with pytest.raises(RuntimeError, match="all of them spent thinking"):
        ask(client)
    # at the most Settings allows, there's no room to give
    client, got = server(cut_off("A cat", captioner.MAX_TOKENS))
    with pytest.raises(RuntimeError, match="cut off"):
        captioner._ask({**LLM, "max_tokens": captioner.MAX_TOKENS}, client, "x", [])
    assert len(got) == 1


def test_a_full_context_is_reported_without_asking_again():
    client, got = server(cut_off("A cat sits on a", 40, prompt=4056))
    with pytest.raises(RuntimeError, match="context is full .*4056"):
        ask(client)
    assert len(got) == 1


def test_server_errors_say_what_the_server_said():
    client, _ = server(400)
    with pytest.raises(httpx.HTTPStatusError, match="400: the request exceeds the available context size"):
        ask(client)


def test_frames_fall_back_to_a_contact_sheet_when_refused(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x96:rate=24:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True)
    client, got = server(400, "A test pattern.")
    assert captioner.caption_file(str(clip), {**LLM, "video_frames": 3}, client) == "A test pattern."
    images = [sum(c.get("type") == "image_url" for c in g["messages"][0]["content"]) for g in got]
    assert images == [3, 1]


def test_a_style_with_a_system_prompt_sends_it_first_with_its_own_sampling(tmp_path):
    from dainamo.server import db

    style = db.FLUX3_CAPTION_STYLE
    cfg = captioner.style_config(LLM, style)
    assert cfg["prompt"] == style["prompt"] and cfg["system"] == style["system"]
    assert (cfg["temperature"], cfg["max_tokens"], cfg["endpoint"]) == (0, 1024, LLM["endpoint"])
    photo = tmp_path / "photo.png"
    Image.new("RGB", (64, 48), "green").save(photo)
    client, got = server("Full-color outdoor photograph,\n a green field.")
    assert captioner.caption_file(str(photo), cfg, client) == "Full-color outdoor photograph, a green field."
    (body,) = got
    system, user = body["messages"]
    assert system == {"role": "system", "content": style["system"]}
    # the image, then the text: how FLUX 3's upsampler gets its references
    assert user["role"] == "user" and [c["type"] for c in user["content"]] == ["image_url", "text"]
    assert user["content"][1]["text"] == style["prompt"]
    assert (body["temperature"], body["max_tokens"]) == (0, 1024)


def test_a_style_without_its_own_settings_keeps_the_providers():
    plain = {"id": "tags", "name": "Tags", "prompt": "Tags."}
    cfg = captioner.style_config(LLM, plain)
    assert (cfg["prompt"], cfg["system"], cfg["temperature"], cfg["max_tokens"]) == ("Tags.", "", 0.2, 300)
    client, got = server("cat, mat")
    assert captioner._ask(cfg, client, cfg["prompt"], [Image.new("RGB", (32, 32))]) == "cat, mat"
    (message,) = got[0]["messages"]  # one user message, its text first
    assert message["role"] == "user" and [c["type"] for c in message["content"]] == ["text", "image_url"]
    assert (got[0]["temperature"], got[0]["max_tokens"]) == (0.2, 300)

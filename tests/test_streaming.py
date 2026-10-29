# Copyright (c) 2026 InBoost Team
# SPDX-License-Identifier: MIT

import unittest

from agent_tool_parser import (
    StreamEvent,
    StreamingToolParser,
)


class TestStreamingToolParser(unittest.TestCase):
    def test_streaming_plain_text(self):
        parser = StreamingToolParser()
        events = parser.feed("Hello world! How are you?")
        events += parser.finish()

        text_events = [e for e in events if e.is_text]
        self.assertTrue(len(text_events) > 0)
        self.assertEqual("".join(e.content for e in text_events), "Hello world! How are you?")

    def test_streaming_thinking_block(self):
        parser = StreamingToolParser()
        events = parser.feed("<think>Analyzing user request...</think>Answer: 42")
        events += parser.finish()

        thinking_events = [e for e in events if e.is_thinking]
        text_events = [e for e in events if e.is_text]

        self.assertEqual("".join(e.content for e in thinking_events), "Analyzing user request...")
        self.assertEqual("".join(e.content for e in text_events), "Answer: 42")

    def test_streaming_dsml_tool_call(self):
        parser = StreamingToolParser()
        chunks = [
            "Let me check that for you.\n",
            "<｜DSML｜tool_calls>\n",
            "<｜DSML｜tool name=search>\n",
            "<｜DSML｜parameter name=pattern string=true>agent</｜DSML｜parameter>\n",
            "<｜DSML｜parameter name=path string=true>src</｜DSML｜parameter>\n",
            "</｜DSML｜tool>\n",
            "</｜DSML｜tool_calls>\nDone!",
        ]
        events = []
        for chunk in chunks:
            events.extend(parser.feed(chunk))
        events.extend(parser.finish())

        started = [e for e in events if e.event_type == "tool_call_started"]
        chunks_arg = [e for e in events if e.event_type == "tool_call_arguments_chunk"]
        completed = [e for e in events if e.event_type == "tool_call_completed"]

        self.assertEqual(len(started), 1)
        self.assertEqual(started[0].tool_name, "search")
        self.assertTrue(len(chunks_arg) > 0)
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].tool_call.name, "search")
        self.assertEqual(
            completed[0].tool_call.args,
            {"pattern": "agent", "path": "src"},
        )

    def test_streaming_markdown_json_tool_call(self):
        parser = StreamingToolParser()
        stream_text = """```json
{
  "name": "read_file",
  "arguments": {
    "path": "README.md"
  }
}
```"""
        events = []
        # Feed byte/character chunks
        chunk_size = 8
        for i in range(0, len(stream_text), chunk_size):
            events.extend(parser.feed(stream_text[i : i + chunk_size]))
        events.extend(parser.finish())

        completed = [e for e in events if e.event_type == "tool_call_completed"]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].tool_call.name, "read_file")
        self.assertEqual(completed[0].tool_call.args, {"path": "README.md"})

    def test_streaming_truncated_finish_recovery(self):
        parser = StreamingToolParser()
        # Incomplete closing tag at EOF
        parser.feed("<｜DSML｜tool name=calc><｜DSML｜parameter name=expr>1 + 1")
        events = parser.finish()

        completed = [e for e in events if e.event_type == "tool_call_completed"]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].tool_call.name, "calc")
        self.assertEqual(completed[0].tool_call.args, {"expr": "1 + 1"})

    def test_stream_event_helpers(self):
        ev_text = StreamEvent(event_type="text", content="hello")
        self.assertTrue(ev_text.is_text)
        self.assertFalse(ev_text.is_thinking)
        self.assertFalse(ev_text.is_tool_call)

        ev_think = StreamEvent(event_type="thinking", content="reasoning")
        self.assertTrue(ev_think.is_thinking)
        self.assertFalse(ev_think.is_text)

        ev_start = StreamEvent(event_type="tool_call_started", tool_name="bash")
        self.assertTrue(ev_start.is_tool_call)

        ev_done = StreamEvent(event_type="tool_call_completed", tool_name="bash")
        self.assertTrue(ev_done.is_tool_call)


if __name__ == "__main__":
    unittest.main()

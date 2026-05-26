import unittest

from agents.structured_output import format_specialist_context, parse_specialist_output


class Phase3StructuredOutputTests(unittest.TestCase):
    def test_parse_specialist_output_normalizes_content_parts_and_adds_raw_text(self):
        content = [{
            "type": "text",
            "text": (
                '{"summary":"Healthy business","key_facts":["Revenue grew 20%"],'
                '"risks":["High valuation"],"sources":["yfinance"],'
                '"data_quality":"good","confidence":"high"}'
            ),
            "extras": {"signature": "ignored"},
        }]

        parsed = parse_specialist_output(content)

        self.assertEqual(parsed["summary"], "Healthy business")
        self.assertEqual(parsed["key_facts"], ["Revenue grew 20%"])
        self.assertEqual(parsed["risks"], ["High valuation"])
        self.assertEqual(parsed["sources"], ["yfinance"])
        self.assertEqual(parsed["data_quality"], "good")
        self.assertEqual(parsed["confidence"], "high")
        self.assertIn('"summary":"Healthy business"', parsed["raw_text"])

    def test_parse_specialist_output_falls_back_for_malformed_json(self):
        parsed = parse_specialist_output("This is not JSON, but it contains useful context.")

        self.assertEqual(parsed["summary"], "This is not JSON, but it contains useful context.")
        self.assertEqual(parsed["key_facts"], [])
        self.assertEqual(parsed["risks"], [])
        self.assertEqual(parsed["sources"], [])
        self.assertEqual(parsed["data_quality"], "weak")
        self.assertEqual(parsed["confidence"], "low")
        self.assertEqual(parsed["raw_text"], "This is not JSON, but it contains useful context.")

    def test_format_specialist_context_includes_quality_confidence_and_raw_fallback(self):
        context = format_specialist_context(
            "Fundamental",
            {
                "summary": "Strong margins.",
                "key_facts": ["Net margin above peers."],
                "risks": [],
                "sources": [],
                "data_quality": "partial",
                "confidence": "medium",
                "raw_text": "Original details.",
            },
        )

        self.assertIn("--- FUNDAMENTAL DATA ---", context)
        self.assertIn("Data quality: partial", context)
        self.assertIn("Confidence: medium", context)
        self.assertIn("Summary: Strong margins.", context)
        self.assertIn("- Net margin above peers.", context)
        self.assertIn("Risks/Caveats:\n- None provided.", context)
        self.assertIn("Sources:\n- None provided.", context)
        self.assertIn("Raw fallback/context:\nOriginal details.", context)


if __name__ == "__main__":
    unittest.main()

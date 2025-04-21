import unittest
from helpers.Renamer import Renamer


class TestRenamer(unittest.TestCase):
	def setUp(self):
		self.renamer = Renamer()
		self.chars = self.renamer.rename_dict

	def test_rename_special_characters(self):
		"""
		Test renaming with special characters
		"""
		for char, replacement in self.chars.items():
			src = f"hello{char}world"
			expected = f"hello{replacement}world"
			renamer = Renamer()
			self.assertEqual(renamer.rename(src), expected)
			del renamer

	def test_rename_spaces(self):
		"""
		Test renaming with spaces
		"""
		self.assertEqual(self.renamer.rename("hello world"), "hello_world")
		self.assertEqual(
			self.renamer.rename("  leading and trailing spaces  "),
			"leading_and_trailing_spaces",
		)

	def test_rename_duplicates(self):
		"""
		Test handling of duplicate names
		"""
		self.assertEqual(self.renamer.rename("duplicate"), "duplicate")
		self.assertEqual(self.renamer.rename("duplicate"), "duplicate_2")
		self.assertEqual(self.renamer.rename("duplicate"), "duplicate_3")

	def test_rename_empty_string(self):
		"""
		Test renaming an empty string
		"""
		self.assertEqual(self.renamer.rename(""), "unnamed")

	def test_rename_only_special_characters(self):
		"""
		Test renaming a string with only special characters
		"""
		src = "!@#$%^&*()"
		expected = f"{self.chars['!']}{self.chars['@']}{self.chars['#']}{self.chars['$']}{self.chars['%']}{self.chars['^']}{self.chars['&']}{self.chars['*']}{self.chars['(']}{self.chars[')']}"
		self.assertEqual(self.renamer.rename(src), expected)

	def test_rename_strip_underscores(self):
		"""
		Test stripping leading and trailing underscores
		"""
		self.assertEqual(self.renamer.rename("__underscore__"), "underscore")

	def test_rename_case_insensitivity(self):
		"""
		Test renaming with mixed case
		"""
		self.assertEqual(self.renamer.rename("HelloWorld"), "helloworld")
		self.assertEqual(self.renamer.rename("HELLO_WORLD"), "hello_world")

	def test_rename_none(self):
		"""
		Test renaming None
		"""
		self.assertEqual(self.renamer.rename(None), "unnamed")


if __name__ == "__main__":
	unittest.main()

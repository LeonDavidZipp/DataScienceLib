from functools import lru_cache


class Renamer:
	"""
	Class for renaming columns to valid python variable names.
	Also handles duplicate names by appending a number to the end of the name.
	"""

	def __init__(self, rename_dict: dict[str, str] | None = None):
		default_rename_dict = {
			"!": "",
			'"': "",
			"#": "number",
			"$": "dollar",
			"%": "percent",
			"&": "and",
			" ": "_",
			"'": "",
			"(": "",
			")": "",
			"*": "star",
			"+": "plus",
			",": "-",
			"-": "_",
			".": "_",
			"/": "_",
			":": "_",
			";": "_",
			"<": "lessthan",
			"=": "equals",
			">": "greaterthan",
			"?": "question",
			"@": "at",
			"[": "_",
			"\\": "_",
			"]": "_",
			"^": "caret",
			"`": "_",
			"{": "_",
			"|": "pipe",
			"}": "_",
			"~": "tilde",
		}
		self.rename_dict = default_rename_dict if rename_dict is None else rename_dict
		self.counts: dict[str, int] = {}

	def rename(self, name: str) -> str:
		"""
		Renames the given name to a valid python variable name by removing special
		characters and replacing spaces with underscores.

		Args:
		        name (str): the name to rename
		Returns:
		        str: the renamed name
		"""
		name = self.rename_inner_(name)
		if self.counts.get(name) is None:
			self.counts[name] = 1
		else:
			self.counts[name] += 1
			name = f"{name}_{self.counts.get(name)}"
		return name

	@lru_cache(maxsize=1024)
	def rename_inner_(self, name: str) -> str:
		if not name:
			return "unnamed"

		name = name.translate(str.maketrans(self.rename_dict))
		name = name.strip("_").lower()
		if not name:
			return "unnamed"
		else:
			return name

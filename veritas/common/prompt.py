from jinja2 import Environment, FileSystemLoader
from pathlib import Path

from ezmm import MultimodalSequence

from veritas.util.util import load_file

env = Environment(loader=FileSystemLoader(""))


class Prompt(MultimodalSequence):
    """Prompt for (M)LLMs, able to handle media and placeholders."""

    def __init__(self, file_path: str = None, text: str = None, **kwargs):
        """
        :param file_path: Path to the Markdown file containing the prompt
        :param text: Prompt text to use instead of the file
        :param kwargs: Keyword parameters to insert the placeholders within the prompt template
        """
        assert file_path or text
        self.file_path = file_path
        data = text or self.fill_template(**kwargs)
        super().__init__(data)

    def fill_template(self, **kwargs) -> str:
        assert self.file_path
        if Path(self.file_path).suffix == ".j2":
            return self.render_jinja2(**kwargs)
        else:
            return self.format_string(**kwargs)

    def format_string(self, **kwargs) -> str:
        assert self.file_path
        text = load_file(self.file_path)
        return text.format(**kwargs) if kwargs else text

    def render_jinja2(self, **kwargs) -> str:
        assert self.file_path
        template = env.get_template(self.file_path)
        return template.render(**kwargs)

from sphinx.application import Sphinx


master_doc = "README"
extensions = [
    "myst_nb",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
]
project = "vbzero"
exclude_patterns = ["playground", ".pytest_cache"]
napoleon_custom_sections = [("Returns", "params_style")]
plot_formats = [
    ("png", 144),
]
html_theme = "sphinx_rtd_theme"

nb_execution_mode = "cache"
nb_execution_raise_on_error = True
nb_execution_timeout = 60

# Configure autodoc to avoid excessively long fully-qualified names.
add_module_names = False
autodoc_typehints_format = "short"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}


def setup(sphinx: Sphinx) -> None:
    # Prevent execution of jupyter notebooks.
    sphinx.registry.source_suffix.pop(".ipynb", None)

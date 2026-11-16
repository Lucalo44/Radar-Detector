"""Docs tab: renders the Markdown pages in docs/, the same files GitHub shows."""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

# (slug, sidebar title, file), in reading order
PAGES = [
    ("numbers", "What the numbers mean", "numbers.md"),
    ("solver", "How the solver works", "solver.md"),
    ("limits", "Assumptions and limits", "limits.md"),
    ("materials", "Materials", "materials.md"),
    ("validation", "Validation", "validation.md"),
    ("example", "Worked example", "example.md"),
]


def page_path(slug: str) -> Path:
    return DOCS_DIR / dict((s, f) for s, _, f in PAGES)[slug]


class DocsTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.nav = QtWidgets.QListWidget()
        self.nav.setMaximumWidth(220)
        for slug, title, _ in PAGES:
            it = QtWidgets.QListWidgetItem(title)
            it.setData(QtCore.Qt.UserRole, slug)
            self.nav.addItem(it)
        self.nav.currentRowChanged.connect(lambda r: self.show_page(PAGES[r][0]) if r >= 0 else None)
        lay.addWidget(self.nav)
        self.browser = QtWidgets.QTextBrowser()
        self.browser.setOpenLinks(False)
        self.browser.anchorClicked.connect(self._on_link)
        self.browser.setSearchPaths([str(DOCS_DIR)])
        self.browser.document().setDefaultStyleSheet("table{border-collapse:collapse} td,th{padding:3px 8px}")
        self.browser.setStyleSheet("QTextBrowser{padding:10px 18px}")
        lay.addWidget(self.browser, 1)
        self._slug = None
        self.nav.setCurrentRow(0)

    def show_page(self, slug: str, heading: str = ""):
        if slug != self._slug:
            path = page_path(slug)
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                text = f"# Missing page\n\nCould not read `{path}`."
            self.browser.document().setBaseUrl(QtCore.QUrl.fromLocalFile(str(DOCS_DIR) + "/"))
            self.browser.setMarkdown(text)
            self._slug = slug
            row = [s for s, _, _ in PAGES].index(slug)
            if self.nav.currentRow() != row:
                self.nav.blockSignals(True)
                self.nav.setCurrentRow(row)
                self.nav.blockSignals(False)
        if heading:
            self._scroll_to(heading)
        else:
            self.browser.verticalScrollBar().setValue(0)

    def _scroll_to(self, heading: str):
        doc = self.browser.document()
        cur = doc.find(heading, 0, QtGui.QTextDocument.FindCaseSensitively)
        if cur.isNull():
            return
        sb = self.browser.verticalScrollBar()
        sb.setValue(sb.maximum())  # scroll past, then back, so the heading lands at the top
        cur.clearSelection()
        self.browser.setTextCursor(cur)
        self.browser.ensureCursorVisible()

    def _on_link(self, url: QtCore.QUrl):
        name = Path(url.path()).name if url.isLocalFile() or not url.scheme() else ""
        for slug, _, f in PAGES:
            if name == f:
                self.show_page(slug, url.fragment().replace("-", " ") if url.fragment() else "")
                return
        if url.scheme() in ("http", "https", "mailto"):
            QtGui.QDesktopServices.openUrl(url)
        elif url.isLocalFile():
            QtGui.QDesktopServices.openUrl(url)

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from hashlib import sha256
from tempfile import gettempdir
from fnmatch import fnmatch

try:
    import pathspec
except ImportError:
    pathspec = None


INDEX_VERSION = 2
CACHE_DIR = Path(gettempdir()) / "local-coding-agent-index"
MAX_FILE_BYTES = 1_000_000
CHUNK_LINES = 80
CHUNK_OVERLAP_LINES = 12
IGNORED_DIRS = {".git", ".agent", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build"}
TEXT_EXTENSIONS = {".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".scala", ".sh", ".bash", ".zsh", ".sql", ".html", ".css", ".scss", ".md", ".rst", ".txt", ".toml", ".yaml", ".yml", ".json", ".xml", ".ini", ".cfg", ".dockerfile"}
TEXT_FILENAMES = {"dockerfile", "makefile", "rakefile", "justfile"}
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")


@dataclass(frozen=True)
class CodeChunk:
    path: str
    start_line: int
    end_line: int
    content: str


class CodebaseIndex:
    """A persistent, line-aware BM25 index rooted at one project directory."""

    def __init__(
        self, 
        root: str | Path
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"Codebase root is not a directory: {self.root}")
        
        key = sha256(str(self.root).encode("utf-8")).hexdigest()
        self.cache_path = CACHE_DIR / f"{key}.json"
        self.chunks: list[CodeChunk] = []
        self._fingerprints: dict[str, list[int]] = {}
        self._gitignore = self._load_gitignore()


    @staticmethod
    def _tokens(text: str) -> list[str]:
        normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text).replace("_", " ")
        return [token.lower() for token in TOKEN_RE.findall(normalized)]


    def _is_indexable(self, path: Path) -> bool:
        relative = path.relative_to(self.root)
        if any(part in IGNORED_DIRS for part in relative.parts):
            return False
        
        if any(part.startswith(".env") for part in relative.parts):
            return False
        
        if self._is_gitignored(relative.as_posix()):
            return False
        
        return path.name.lower() in TEXT_FILENAMES or path.suffix.lower() in TEXT_EXTENSIONS


    def _load_gitignore(self):
        gitignore = self.root / ".gitignore"
        if not gitignore.is_file():
            return None
        
        try:
            lines = gitignore.read_text(encoding="utf-8").splitlines()
            if pathspec is not None:
                return pathspec.PathSpec.from_lines("gitwildmatch", lines)
            
            return [
                line.strip() 
                for line in lines 
                if line.strip() and not line.lstrip().startswith("#")
            ]
        
        except OSError:
            return None

    def _is_gitignored(self, relative: str) -> bool:
        if not self._gitignore:
            return False
        
        if pathspec is not None and hasattr(self._gitignore, "match_file"):
            return self._gitignore.match_file(relative)
        
        ignored = False
        for pattern in self._gitignore:
            negated = pattern.startswith("!")
            pattern = pattern[1:] if negated else pattern
            pattern = pattern.lstrip("/")
            matches = (
                fnmatch(relative, pattern)
                or fnmatch(Path(relative).name, pattern)
                or (pattern.endswith("/") and relative.startswith(pattern))
            )
            
            if matches:
                ignored = not negated
                
        return ignored


    def _files_and_fingerprints(
        self
    ) -> tuple[list[Path], dict[str, list[int]]]:
        files: list[Path] = []
        fingerprints: dict[str, list[int]] = {}
        gitignore = self.root / ".gitignore"
        if gitignore.is_file():
            try:
                stat = gitignore.stat()
                fingerprints[".gitignore"] = [stat.st_mtime_ns, stat.st_size]
            except OSError:
                pass
            
        for path in self.root.rglob("*"):
            if not path.is_file() or not self._is_indexable(path):
                continue
            
            try:
                stat = path.stat()
            except OSError:
                continue
            
            if stat.st_size > MAX_FILE_BYTES:
                continue
            
            relative = path.relative_to(self.root).as_posix()
            files.append(path)
            fingerprints[relative] = [stat.st_mtime_ns, stat.st_size]
            
        return files, fingerprints


    def _load_if_fresh(
        self, 
        fingerprints: dict[str, list[int]]
    ) -> bool:
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if payload.get("version") != INDEX_VERSION or payload.get("fingerprints") != fingerprints:
                return False
            self.chunks = [CodeChunk(**chunk) for chunk in payload["chunks"]]
            self._fingerprints = fingerprints
            return True
        
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_VERSION, 
            "fingerprints": self._fingerprints, 
            "chunks": [asdict(chunk) for chunk in self.chunks]
        }
        self.cache_path.write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8"
        )


    def build(
        self, 
        force: bool = False
    ) -> bool:
        """Build or refresh the index. Returns True when files were re-indexed."""
        files, fingerprints = self._files_and_fingerprints()
        if not force and self._load_if_fresh(fingerprints):
            return False
        
        chunks: list[CodeChunk] = []
        for path in files:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            
            relative = path.relative_to(self.root).as_posix()
            for start in range(0, max(len(lines), 1), CHUNK_LINES - CHUNK_OVERLAP_LINES):
                selected = lines[start : start + CHUNK_LINES]
                if not selected:
                    continue
                
                chunks.append(CodeChunk(relative, start + 1, start + len(selected), "\n".join(selected)))
                if start + CHUNK_LINES >= len(lines):
                    break
                
        self.chunks = chunks
        self._fingerprints = fingerprints
        self._save()
        return True


    def search(
        self, 
        query: str, 
        limit: int = 5
    ) -> list[tuple[CodeChunk, float]]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        
        self.build()
        query_tokens = self._tokens(query)
        if not query_tokens or not self.chunks:
            return []
        
        documents = [Counter(self._tokens(chunk.content)) for chunk in self.chunks]
        count = len(documents)
        average_length = sum(sum(doc.values()) for doc in documents) / count
        frequency = Counter(token for doc in documents for token in set(doc))
        query_frequency = Counter(query_tokens)
        scores: list[tuple[int, float]] = []
        
        for index, doc in enumerate(documents):
            length, score = sum(doc.values()), 0.0
            for token, query_count in query_frequency.items():
                term_frequency = doc.get(token, 0)
                if not term_frequency:
                    continue
                
                inverse_frequency = math.log(1 + (count - frequency[token] + 0.5) / (frequency[token] + 0.5))
                score += query_count * inverse_frequency * (term_frequency * 2.5) / (term_frequency + 1.5 * (0.25 + 0.75 * length / max(average_length, 1)))
                
            if score:
                path_tokens = set(self._tokens(self.chunks[index].path))
                scores.append((index, score + 0.15 * len(path_tokens.intersection(query_frequency))))
                
        return [
            (self.chunks[index], score) 
            for index, score in sorted(
                scores, key=lambda item: item[1], reverse=True
            )[:limit]
        ]

    def format_results(
        self, 
        query: str, 
        limit: int = 5, 
        max_chars: int = 12_000
    ) -> str:
        results = self.search(query, limit)
        if not results:
            return "(no relevant indexed code found)"
        
        sections: list[str] = []
        used = 0
        for chunk, score in results:
            section = f"### {chunk.path}:{chunk.start_line}-{chunk.end_line} (relevance {score:.2f})\n```\n{chunk.content}\n```"
            if used and used + len(section) > max_chars:
                break
            
            sections.append(section)
            used += len(section)

        return "\n\n".join(sections)

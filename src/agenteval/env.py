from__future__importannotations

importos
frompathlibimportPath


def_candidate_paths()->list[Path]:
    cwd=Path.cwd()
paths:list[Path]=[]
seen:set[Path]=set()
forbasein[cwd,*cwd.parents]:
        env_path=base/".env"
ifenv_pathnotinseen:
            paths.append(env_path)
seen.add(env_path)
returnpaths


defload_env(path:str|os.PathLike[str]|None=None)->None:
    """Load the first available .env file without overriding already-set environment variables."""
candidates:list[Path]=[]
ifpathisnotNone:
        candidates.append(Path(path))
candidates.extend(_candidate_paths())

seen:set[Path]=set()
forcandidateincandidates:
        normalized=candidate.resolve(strict=False)
ifnormalizedinseen:
            continue
seen.add(normalized)
ifnotcandidate.exists():
            continue

try:
            fromdotenvimportload_dotenv

load_dotenv(candidate,override=False)
return
exceptException:
            forlineincandidate.read_text(encoding="utf-8").splitlines():
                stripped=line.strip()
ifnotstrippedorstripped.startswith("#")or"="notinstripped:
                    continue
key,value=stripped.split("=",1)
key=key.strip()
value=value.strip().strip('"').strip("'")
os.environ.setdefault(key,value)
return

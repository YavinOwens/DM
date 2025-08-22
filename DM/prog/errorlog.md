This errorlog, is brain dumps of how issues where resolved. Vibe coding seems not to be as effective when using online services i.e Codespaces. (May research why later and update)

1st run 
codespace doesnt pick up local instance of ollama
Fix : Remote IP <home>11434/...
i.e : OLLAMA_API_URL = "http://123.45.67.89:11434/api/generate"


codespace didnt pick up local models, 
Fix curl -fsSL https://ollama.com/install.sh | sh , ollama serve, ollama pull <model>
(might add a local version of fix)

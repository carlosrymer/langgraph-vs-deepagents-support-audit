# Third-party notices

This project builds on **LangGraph** (`langchain-ai/langgraph`) and **Deep Agents**
(`langchain-ai/deepagents`), © LangChain, Inc., both MIT-licensed. They are installed as
dependencies at `langgraph==1.2.12` and `deepagents==0.7.19` (with `langchain==1.4.2`,
`langchain-core==1.6.5` and `langchain-openai==1.6.6`); none of them is vendored or forked.

Nothing is vendored. The only upstream text in this repository is short identifiers quoted in
docs and code comments (middleware names, default thresholds, the `/large_tool_results/` path),
cited to explain behaviour I measured.

The upstream licenses ship with the installed packages (see each package's `LICENSE` in its
PyPI distribution); I don't redistribute their code, so I don't reproduce the license text here.

The support-ops corpus, its generator, the eight tasks and their oracle, the grader, the metering
and reporting code, the verifier, the offline scripted model and the site are my own work.

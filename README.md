# Claude + AgentFS + LlamaIndex Workflows

A demo where we run Claude Code within a fully-virtualized file system ([AgentFS](https://github.com/tursodatabase/agentfs)), orchestrating it with [LlamaIndex Workflows](https://github.com/run-llama/workflows-ts) and adding the possibility of reading unstructured files (e.g. PDFs or Word/Google docs) with [LlamaCloud](https://cloud.llamaindex.ai).

## Set Up and Run

Clone this repository:

```bash
git clone https://github.com/run-llama/agentfs-claude
cd agentfs-claude
```

Install all necessary dependencies:

```bash
pnpm install 
# you can use other package managers, but pnpm is preferred
```

Now run the demo with:

```bash
# for the first time
pnpm run start

# for follow-ups
pnpm run clean-start
```

And follow the prompts in the terminal

## How it works

All the [filesystem-bound operations](./src/filesystem.ts) are performed on the AgentFS (and not on the real files), thanks to the [`filesystem` MCP](./src/mcp.ts) that exposes the following tools:

- `read_file`: read a file, providing its path
- `write_file`: write a file, providing its path and content
- `edit_file`: edit a file, providing the old string and the new string to replace the old one with
- `list_files`: list all the available files
- `file_exists`: check whether or not a file exists, providing its path

What happens under the hood when the agent is running:

- All text-based files in the current directory are uploaded to a [LibSQL](https://github.com/tursodatabase/libsql) database and indexed
- Non text-based files (namely in PDF/DOCX/DOC/PPTX/XLSX format) are parsed by [LlamaParse](https://cloud.llamaindex.ai) and uploaded with their content converted to markdown text
- When the agent performs a filesystem-bound operation, it calls one of the tools from the `filesystem` MCP
- Other tools (such as WebSearch/Todo/Task) run normally
- If the agent was to call one of the disallowed tools (Read, Write, Edit and Glob), a `PreToolUse` hook would deny the tool call and redirect the agent to using the `filesystem` MCP tools

The integration with LlamaIndex Workflows offers the perfect harnessed environment:

- Files are pre-loaded from the current directory into AgentFS in the first step of the workflow
- The prompt from the user and other information (resume a session, use plan mode or not) are collected from the user directly using human-in-the-loop
- The agent runs in its own step, when everything is ready

## Contributing

If you wish to contribute, make sure that your code follows the formatting and linting guidelines, running:

```bash
pnpm run check
```

Once your code is compliant with formatting and linting, you can create a pull request from a non-default branch of your fork (e.g. `feat/awesome-feature`, `fix/great-fix`).

## License

This project is distributed under an [MIT License](./LICENSE).
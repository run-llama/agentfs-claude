import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { type CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import {
  readSchemaShape,
  fileExistsSchemaShape,
  writeSchemaShape,
  listFilesSchemaShape,
  editSchemaShape,
  getAgentFS,
} from "./mcp";
import {
  readFile,
  writeFile,
  editFile,
  fileExists,
  listFiles,
} from "./filesystem";

const mcpServer = new McpServer({
  name: "filesystem-mcp",
  version: "1.0.0",
});

mcpServer.registerTool(
  "read_file",
  {
    description: "Read a file by passing its path.",
    inputSchema: readSchemaShape,
  },
  async ({ filePath }) => {
    const agentfs = await getAgentFS({});
    const content = await readFile(filePath, agentfs);
    if (typeof content == "string") {
      return { content: [{ type: "text", text: content }] };
    } else {
      return {
        content: [
          {
            type: "text",
            text: `Could not read ${filePath}. Please check that the file exists and submit the request again.`,
          },
        ],
        isError: true,
      };
    }
  },
);

mcpServer.registerTool(
  "file_exists",
  {
    description: "Check whether a file exists or not by passing its path.",
    inputSchema: fileExistsSchemaShape,
  },
  async ({ filePath }) => {
    const agentfs = await getAgentFS({});
    const exists = await fileExists(filePath, agentfs);
    if (exists) {
      return {
        content: [{ type: "text", text: `File ${filePath} exists` }],
      };
    } else {
      return {
        content: [{ type: "text", text: `File ${filePath} does not exist.` }],
      };
    }
  },
);

mcpServer.registerTool(
  "write_file",
  {
    description: "Write a file by passing its path and content.",
    inputSchema: writeSchemaShape,
  },
  async ({ filePath, fileContent }) => {
    const agentfs = await getAgentFS({});
    const success = await writeFile(filePath, fileContent, agentfs);
    if (success) {
      return {
        content: [
          {
            type: "text",
            text: `File ${filePath} successfully written with content:\n\n'''\n${fileContent}\n'''`,
          },
        ],
      };
    } else {
      return {
        content: [
          {
            type: "text",
            text: `There was an error while writing file ${filePath}`,
          },
        ],
      };
    }
  },
);

mcpServer.registerTool(
  "edit_file",
  {
    description:
      "Edit a file by passing its path, the old string and the new string.",
    inputSchema: editSchemaShape,
  },
  async ({ filePath, oldString, newString }) => {
    const agentfs = await getAgentFS({});
    const editedContent = await editFile(
      filePath,
      oldString,
      newString,
      agentfs,
    );
    if (typeof editedContent == "string") {
      return {
        content: [
          {
            type: "text",
            text: `Successfully edited ${filePath}. New content:\n\n'''\n${editedContent}\n'''`,
          },
        ],
      } as CallToolResult;
    } else {
      return {
        content: [
          {
            type: "text",
            text: `Could not edit ${filePath}. Please check that the file exists and submit the request again.`,
          },
        ],
        isError: true,
      };
    }
  },
);

mcpServer.registerTool(
  "list_files",
  {
    description: "List all the available files",
    inputSchema: listFilesSchemaShape,
  },
  async () => {
    const agentfs = await getAgentFS({});
    const files = await listFiles(agentfs);
    if (files != "") {
      return { content: [{ type: "text", text: files }] };
    } else {
      return {
        content: [
          {
            type: "text",
            text: `Could not list files. Please report this failure to the user`,
          },
        ],
        isError: true,
      };
    }
  },
);

async function main() {
  const transport = new StdioServerTransport();
  await mcpServer.connect(transport);
  console.log("MCP server is running...");
}

main().catch((error) => {
  console.error("Server error:", error);
  process.exit(1);
});

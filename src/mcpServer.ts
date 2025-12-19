import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { Request, Response } from "express";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import {
  type CallToolResult,
  isInitializeRequest,
} from "@modelcontextprotocol/sdk/types.js";
import { randomUUID } from "node:crypto";
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
import { createMcpExpressApp } from "@modelcontextprotocol/sdk/server/express.js";

const getServer = () => {
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
      const agentfs = await getAgentFS({ filePath: "fsMcp.db" });
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
      const agentfs = await getAgentFS({ filePath: "fsMcp.db" });
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
      const agentfs = await getAgentFS({ filePath: "fsMcp.db" });
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
      const agentfs = await getAgentFS({ filePath: "fsMcp.db" });
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
      const agentfs = await getAgentFS({ filePath: "fsMcp.db" });
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
  return mcpServer;
};

const app = createMcpExpressApp();

const transports: { [sessionId: string]: StreamableHTTPServerTransport } = {};

app.post("/mcp", async (req: Request, res: Response) => {
  console.log("Received MCP request:", req.body);
  try {
    // Check for existing session ID
    const sessionId = req.headers["mcp-session-id"] as string | undefined;
    let transport: StreamableHTTPServerTransport;

    if (sessionId && transports[sessionId]) {
      // Reuse existing transport
      transport = transports[sessionId];
    } else if (!sessionId && isInitializeRequest(req.body)) {
      // New initialization request - use JSON response mode
      transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: () => randomUUID(),
        enableJsonResponse: true, // Enable JSON response mode
        onsessioninitialized: (sessionId) => {
          // Store the transport by session ID when session is initialized
          // This avoids race conditions where requests might come in before the session is stored
          console.log(`Session initialized with ID: ${sessionId}`);
          transports[sessionId] = transport;
        },
      });

      // Connect the transport to the MCP server BEFORE handling the request
      const server = getServer();
      await server.connect(transport);
      await transport.handleRequest(req, res, req.body);
      return; // Already handled
    } else {
      // Invalid request - no session ID or not initialization request
      res.status(400).json({
        jsonrpc: "2.0",
        error: {
          code: -32000,
          message: "Bad Request: No valid session ID provided",
        },
        id: null,
      });
      return;
    }

    // Handle the request with existing transport - no need to reconnect
    await transport.handleRequest(req, res, req.body);
  } catch (error) {
    console.error("Error handling MCP request:", error);
    if (!res.headersSent) {
      res.status(500).json({
        jsonrpc: "2.0",
        error: {
          code: -32603,
          message: "Internal server error",
        },
        id: null,
      });
    }
  }
});

// Handle GET requests for SSE streams according to spec
app.get("/mcp", async (req: Request, res: Response) => {
  // Since this is a very simple example, we don't support GET requests for this server
  // The spec requires returning 405 Method Not Allowed in this case
  res.status(405).set("Allow", "POST").send("Method Not Allowed");
});

// Start the server
const PORT = 3000;
app.listen(PORT, (error) => {
  if (error) {
    console.error("Failed to start server:", error);
    process.exit(1);
  }
  console.log(`MCP Streamable HTTP Server listening on port ${PORT}`);
});

// Handle server shutdown
process.on("SIGINT", async () => {
  console.log("Shutting down server...");
  process.exit(0);
});

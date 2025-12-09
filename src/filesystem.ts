import * as fs from "fs/promises";
import { type FileWithContent } from "./types";
import path from "path";
import * as mime from "mime-types";
import { AgentFS } from "agentfs-sdk";
import { parseFile } from "./llamacloud";

const commonCodeExtenstions = [
  ".py",
  ".js",
  ".ts",
  ".c",
  ".cpp",
  ".rs",
  ".go",
  ".java",
  ".cs",
  ".json",
  ".yaml",
  ".sql",
  ".zig",
  ".tsx",
  ".jsx",
  ".html",
  ".css",
  ".pl",
  ".php",
  ".rb",
  ".r",
];

export const filesToParseExtensions = [
  ".pdf",
  ".docx",
  ".doc",
  ".pptx",
  ".xlsx",
];

async function getFilesInDir({
  dirPath = "./",
  toIgnore = ["node_modules", "dist", "pnpm-lock.yaml", ".git"],
}: {
  dirPath?: string;
  toIgnore?: string[];
}): Promise<FileWithContent[]> {
  const files: FileWithContent[] = [];
  try {
    const entries = await fs.readdir(dirPath, { withFileTypes: true });
    for (const entry of entries) {
      const pt = path.join(dirPath, entry.name);
      let toSkip = false;
      for (const dirToIgnore of toIgnore) {
        if (pt.includes(dirToIgnore)) {
          toSkip = true;
        }
      }
      if (toSkip) {
        continue;
      }
      if (entry.isDirectory()) {
        const subFiles = await getFilesInDir({ dirPath: pt });
        files.push(...subFiles);
      } else {
        const mimeType = mime.lookup(pt);
        const ext = "." + pt.split(".").at(-1);
        if (
          (typeof mimeType === "string" && mimeType.startsWith("text/")) ||
          commonCodeExtenstions.includes(ext)
        ) {
          const content = await fs.readFile(pt, { encoding: "utf-8" });
          files.push({ filePath: pt, content: content });
        } else if (filesToParseExtensions.includes(ext)) {
          const content = await parseFile(pt);
          files.push({ filePath: pt, content: content });
        }
      }
    }
  } catch (error) {
    console.error(error);
  }
  return files;
}

export async function recordFiles(
  agentfs: AgentFS,
  {
    dirPath = "./",
    toIgnore = ["node_modules", "dist", "pnpm-lock.yaml", ".git"],
  }: { dirPath?: string; toIgnore?: string[] },
): Promise<boolean> {
  try {
    const files = await getFilesInDir({ dirPath: dirPath, toIgnore: toIgnore });
    for (const file of files) {
      if (path.isAbsolute(file.filePath)) {
        await agentfs.fs.writeFile(file.filePath, file.content);
      } else {
        if (file.filePath.startsWith("./")) {
          file.filePath = file.filePath.replace("./", "/");
          await agentfs.fs.writeFile(file.filePath, file.content);
        } else {
          file.filePath = "/" + file.filePath;
          await agentfs.fs.writeFile(file.filePath, file.content);
        }
      }
    }
    return true;
  } catch (error) {
    console.error(error);
    return false;
  }
}

export async function readFile(
  filePath: string,
  agentfs: AgentFS,
): Promise<string | null> {
  let content: string | null = null;
  try {
    content = (await agentfs.fs.readFile(filePath, "utf-8")) as string;
  } catch (error) {
    console.error(error);
  }
  return content;
}

export async function writeFile(
  filePath: string,
  fileContent: string,
  agentfs: AgentFS,
): Promise<boolean> {
  if (!filePath.startsWith("/")) {
    filePath = "/" + filePath;
  }
  try {
    await agentfs.fs.writeFile(filePath, fileContent);
    return true;
  } catch (error) {
    console.error(error);
    return false;
  }
}

export async function editFile(
  filePath: string,
  oldString: string,
  newString: string,
  agentfs: AgentFS,
): Promise<string | null> {
  if (!filePath.startsWith("/")) {
    filePath = "/" + filePath;
  }
  let editedContent: string | null = null;
  try {
    const content = (await agentfs.fs.readFile(filePath, "utf-8")) as string;
    editedContent = content.replace(oldString, newString);
    await agentfs.fs.writeFile(filePath, editedContent);
    return editedContent;
  } catch (error) {
    console.error(error);
    return editedContent;
  }
}

export async function fileExists(
  filePath: string,
  agentfs: AgentFS,
): Promise<boolean> {
  try {
    if (!filePath.startsWith("/")) {
      filePath = "/" + filePath;
    }
    const dirPath = path.dirname(filePath);
    const files = await agentfs.fs.readdir(dirPath);
    for (const file of files) {
      if (file == filePath || file == path.basename(filePath)) {
        return true;
      }
    }
    return false;
  } catch (error) {
    console.error(error);
    return false;
  }
}

function isDir(path: string): boolean {
  const sp = path.split(".");
  return sp.length == 1;
}

export async function listFiles(agentfs: AgentFS): Promise<string> {
  const dirPath = "/";
  let availableFiles: string = "";
  try {
    const files = await agentfs.fs.readdir(dirPath);
    availableFiles += "AVAILABLE FILES:\n";
    for (const file of files) {
      if (isDir(file)) {
        const subFiles = await agentfs.fs.readdir(path.join(dirPath, file));
        for (const subFile of subFiles) {
          files.push(path.join(file, subFile));
        }
      } else {
        availableFiles += file + ", ";
      }
    }
    return availableFiles.trim();
  } catch (error) {
    console.error(error);
    return availableFiles;
  }
}

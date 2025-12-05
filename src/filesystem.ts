import * as fs from 'fs/promises';
import {type FileWithContent} from "./types";
import path from 'path';
import * as mime from 'mime-types';
import { AgentFS } from 'agentfs-sdk';

async function getFilesInDir({dirPath = "./"}: {dirPath?: string}): Promise<FileWithContent[]> {
    const files: FileWithContent[] = []
    try {
        const entries = await fs.readdir(dirPath, { withFileTypes: true })
        for (const entry of entries) {
            const pt = path.join(dirPath, entry.name)
            if (entry.isDirectory()) {
                const subFiles = await getFilesInDir({dirPath: pt})
                files.push(...subFiles)
            } else {
                const mimeType = mime.lookup(pt)
                if (typeof mimeType === "string" && mimeType.startsWith("text/")) {
                    const content = await fs.readFile(pt, {encoding: "utf-8"})
                    files.push({filePath: pt, content: content})
                }
            }
        }
    } catch(error) {
        console.error(error)
    }
    return files
}

export async function recordFiles(agentfs: AgentFS): Promise<boolean> {
    try {
        const files = await getFilesInDir({})
        for (const file of files) {
            await agentfs.fs.writeFile(file.filePath, file.content)
        }
        return true
    } catch(error) {
        console.error(error)
        return false
    }
} 

export async function readFile(filePath: string, agentfs: AgentFS): Promise<string | null> {
    let content: string | null = null
    try {
        content = await agentfs.fs.readFile(filePath, "utf-8") as string
    } catch(error) {
        console.error(error)
    }
    return content
}

export async function writeFile(filePath: string, fileContent: string, agentfs: AgentFS): Promise<boolean> {
    try {
        await agentfs.fs.writeFile(filePath, fileContent)
        return true
    } catch(error) {
        console.error(error)
        return false
    }
}

export async function editFile(filePath: string, oldString: string, newString: string, agentfs: AgentFS): Promise<string | null> {
    let editedContent: string | null = null
    try {
        const content = await agentfs.fs.readFile(filePath, "utf-8") as string
        editedContent = content.replace(oldString, newString)
        await agentfs.fs.writeFile(filePath, editedContent)
        return editedContent
    } catch(error) {
        console.error(error)
        return editedContent
    }
}

export async function fileExists(filePath: string, agentfs: AgentFS): Promise<boolean> {
    try {
        const dirPath = path.dirname(filePath)
        const files = await agentfs.fs.readdir(dirPath)
        for (const file of files) {
            if (file == filePath || file == path.basename(filePath)) {
                return true
            }
        }
        return false
    } catch(error) {
        console.error(error)
        return false
    }
}

export async function listFiles(agentfs: AgentFS): Promise<string> {
    const dirPath = "./"
    let availableFiles: string = ""
    try {
        const files = await agentfs.fs.readdir(dirPath)
        availableFiles += "AVAILABLE FILES:\n"
        for (const file of files) {
            availableFiles += file + ", "
        }
        return availableFiles.trim()
    } catch(error) {
        console.error(error)
        return availableFiles
    }
}
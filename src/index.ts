import { createWorkflow, workflowEvent } from "@llamaindex/workflow-core";
import { createStatefulMiddleware } from "@llamaindex/workflow-core/middleware/state";
import { recordFiles } from "./filesystem";
import { getAgentFS } from "./mcp";
import { Agent } from "./claude";
import { queryOptions } from "./options";
import { bold, green, red } from "@visulima/colorize";
import { consoleInput, renderLogo } from "./cli";
import * as fs from "fs";
import { runCodex } from "./codex";

async function main() {
  const { withState } = createStatefulMiddleware(() => ({}));
  const workflow = withState(createWorkflow());
  const startEvent = workflowEvent<{ workingDirectory: string | undefined }>();
  const filesRegisteredEvent = workflowEvent<void>();
  const requestPromptEvent = workflowEvent<void>();
  const promptEvent = workflowEvent<{
    chosenAgent: string;
    prompt: string;
    resume: string | undefined;
    plan: boolean;
  }>();
  const stopEvent = workflowEvent<{ success: boolean; error: string | null }>();

  const notFromScratch = fs.existsSync("fs.db");

  const agentFs = await getAgentFS({});

  workflow.handle([startEvent], async (_context, event) => {
    await renderLogo();
    if (notFromScratch) {
      fs.copyFileSync("fs.db", "fsMcp.db");
      if (fs.existsSync("fs.db-wal")) {
        fs.copyFileSync("fs.db-wal", "fsMcp.db-wal");
      }
      return filesRegisteredEvent.with();
    }
    const wd = event.data.workingDirectory;
    let dirPath: string | undefined = wd;
    if (typeof wd === "undefined") {
      dirPath = "./";
    }
    const success = await recordFiles(agentFs, { dirPath: dirPath });
    if (!success) {
      return stopEvent.with({
        success: success,
        error:
          "Could not register the files within the AgentFS file system: check writing permissions in the current directory",
      });
    } else {
      fs.copyFileSync("fs.db", "fsMcp.db");
      if (fs.existsSync("fs.db-wal")) {
        fs.copyFileSync("fs.db-wal", "fsMcp.db-wal");
      }
      return filesRegisteredEvent.with();
    }
  });

  // eslint-disable-next-line
  workflow.handle([filesRegisteredEvent], async (_context, _event) => {
    console.log(
      bold(
        green(
          "All the files have been uploaded to the AgentFS filesystem, what would you like to do now?",
        ),
      ),
    );
    return requestPromptEvent.with();
  });

  workflow.handle([promptEvent], async (_context, event) => {
    const prompt = event.data.prompt;
    if (event.data.chosenAgent == "claude") {
      const agent = new Agent(queryOptions, {
        resume: event.data.resume,
        plan: event.data.plan,
      });
      try {
        await agent.run(prompt);
        return stopEvent.with({ success: true, error: null });
      } catch (error) {
        return stopEvent.with({ success: false, error: JSON.stringify(error) });
      }
    } else {
      try {
        await runCodex(event.data.prompt, { resumeSession: event.data.resume });
        return stopEvent.with({ success: true, error: null });
      } catch (error) {
        console.error(error);
        return stopEvent.with({ success: false, error: JSON.stringify(error) });
      }
    }
  });

  const { sendEvent, snapshot, stream } = workflow.createContext();
  sendEvent(startEvent.with({ workingDirectory: "./" }));
  await stream.until(requestPromptEvent).toArray();
  const snapshotData = await snapshot();
  let agentOfChoice = "claude";
  const chosenAgent = await consoleInput(
    "What agent would you like to use? [codex/claude] ",
  );
  if (chosenAgent.trim() != "") {
    agentOfChoice = chosenAgent;
  }
  const humanResponse = await consoleInput("Your prompt: ");
  console.log(
    bold("Would you like to resume a previous session? Leave blank if not"),
  );
  const resumeSession = await consoleInput("Your answer: ");
  let sessionId: string | undefined = undefined;
  if (resumeSession.trim() != "") {
    sessionId = resumeSession;
  }
  let planMode = false;
  if (agentOfChoice == "claude") {
    console.log(bold("Would you like to activate plan mode? [y/n]"));
    const activatePlan = await consoleInput("Your answer: ");
    if (["yes", "y", "yse"].includes(activatePlan.trim().toLowerCase())) {
      planMode = true;
    }
  }
  const resumedContext = workflow.resume(snapshotData);
  resumedContext.sendEvent(
    promptEvent.with({
      chosenAgent: agentOfChoice,
      prompt: humanResponse,
      resume: sessionId,
      plan: planMode,
    }),
  );
  const finalEvent = (
    await resumedContext.stream.until(stopEvent).toArray()
  ).at(-1);
  if (typeof finalEvent != "undefined") {
    if ("error" in finalEvent.data) {
      if (typeof finalEvent.data.error == "string") {
        console.log(
          `${bold(red("An error occurred during the workflow execution:"))} ${finalEvent.data.error}`,
        );
      }
    }
  }
}

await main().catch(console.error);

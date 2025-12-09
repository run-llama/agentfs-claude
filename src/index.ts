import { createWorkflow, workflowEvent } from "@llamaindex/workflow-core";
import { createStatefulMiddleware } from "@llamaindex/workflow-core/middleware/state";
import { recordFiles } from "./filesystem";
import { getAgentFS } from "./mcp";
import { Agent } from "./claude";
import { queryOptions } from "./options";
import { bold } from "@visulima/colorize";
import { consoleInput } from "./cli";
import * as fs from "fs";

async function main() {
  const { withState } = createStatefulMiddleware(() => ({}));
  const workflow = withState(createWorkflow());
  const startEvent = workflowEvent<{ workingDirectory: string | undefined }>();
  const filesRegisteredEvent = workflowEvent<void>();
  const requestPromptEvent = workflowEvent<void>();
  const promptEvent = workflowEvent<{
    prompt: string;
    resume: string | undefined;
    plan: boolean;
  }>();
  const stopEvent = workflowEvent<{ success: boolean; error: string | null }>();

  const agentFs = await getAgentFS({});

  workflow.handle([startEvent], async (_context, event) => {
    if (fs.existsSync("fs.db")) {
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
      return filesRegisteredEvent.with();
    }
  });

  // eslint-disable-next-line
  workflow.handle([filesRegisteredEvent], async (_context, _event) => {
    console.log(
      bold(
        "All the files have been uploaded to the AgentFS filesystem, what would you like to do now?",
      ),
    );
    return requestPromptEvent.with();
  });

  workflow.handle([promptEvent], async (_context, event) => {
    const prompt = event.data.prompt;
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
  });

  const { sendEvent, snapshot, stream } = workflow.createContext();
  sendEvent(startEvent.with({ workingDirectory: "./" }));
  await stream.until(requestPromptEvent).toArray();
  const snapshotData = await snapshot();
  const humanResponse = await consoleInput("Your prompt: ");
  console.log(
    bold("Would you like to resume a previous session? Leave blank if not"),
  );
  const resumeSession = await consoleInput("Your answer: ");
  let sessionId: string | undefined = undefined;
  if (resumeSession.trim() != "") {
    sessionId = resumeSession;
  }
  console.log(bold("Would you like to activate plan mode? [y/n]"));
  const activatePlan = await consoleInput("Your answer: ");
  let planMode = false;
  if (["yes", "y", "yse"].includes(activatePlan.trim().toLowerCase())) {
    planMode = true;
  }
  const resumedContext = workflow.resume(snapshotData);
  resumedContext.sendEvent(
    promptEvent.with({
      prompt: humanResponse,
      resume: sessionId,
      plan: planMode,
    }),
  );
  await resumedContext.stream.until(stopEvent).toArray();
}

await main().catch(console.error);

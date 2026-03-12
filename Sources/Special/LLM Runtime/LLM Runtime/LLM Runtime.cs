using LLama.Common;
using LLama;
using System.Globalization;
using System.Text;
using System.Text.Json;

namespace LLMRuntime;

static class PromptMarkers
{
    public const string System = "<|### System ###|>";
    public const string Assistant = "<|### Assistant ###|>";
    public const string User = "<|### User ###|>";
    public const string Tools = "<|### Tools ###|>";
    public const string Tool = "<|### Tool ###|>";
    public const string ToolCallsPrefix = "<|tool_calls|>";
    public const string ToolNameSeparator = "<|tool_name|>";
}

public class LLMExecutor
{
    readonly double Temperature;
    readonly int MaxTokens;
    readonly IToolCallHandler ToolHandler;

    readonly ModelParams Parameters;
    readonly LLamaWeights Model;
    InteractiveExecutor Executor;

    string? _cachedSystemPromptWithTools;
    string? _cachedToolsJson;

    public LLMExecutor(string modelPath, double temperature, int maxTokens, int gpuLayers, string toolFormat)
    {
        Temperature = temperature;
        MaxTokens = maxTokens;
        ToolHandler = string.IsNullOrWhiteSpace(toolFormat)
            ? ToolCallHandlerFactory.None
            : ToolCallHandlerFactory.Resolve(toolFormat);

        Parameters = new ModelParams(modelPath)
        {
            ContextSize = 32768,
            GpuLayerCount = gpuLayers
        };
        Model = LLamaWeights.LoadFromFile(Parameters);
        Executor = new InteractiveExecutor(Model.CreateContext(Parameters));
    }

    ParsedPrompt ParsePrompt(string rawPrompt)
    {
        var chatHistory = new ChatHistory();
        var toolResults = new List<(string Name, string Result)>();
        var toolJsonBuilder = new StringBuilder();

        AuthorRole role = AuthorRole.Unknown;
        var messageBuffer = new StringBuilder();

        void flushMessage()
        {
            if (role == AuthorRole.Unknown)
                return;

            string content = messageBuffer.ToString();
            messageBuffer.Clear();

            if (role == AuthorRole.User && content.StartsWith(PromptMarkers.ToolNameSeparator, StringComparison.Ordinal))
            {
                int sep = content.IndexOf('\n');
                if (sep > 0)
                {
                    string toolName = content[PromptMarkers.ToolNameSeparator.Length..sep].Trim();
                    toolResults.Add((toolName, content[(sep + 1)..].TrimEnd()));
                }
                else
                {
                    chatHistory.AddMessage(role, content);
                }
            }
            else
            {
                chatHistory.AddMessage(role, content);
            }
        }

        foreach (string line in rawPrompt.Split([Environment.NewLine, "\n"], StringSplitOptions.None))
        {
            string input = line.Trim();

            if (input.StartsWith("<|###", StringComparison.Ordinal))
            {
                flushMessage();

                if (input == PromptMarkers.System)
                    role = AuthorRole.System;
                else if (input == PromptMarkers.Assistant)
                    role = AuthorRole.Assistant;
                else if (input == PromptMarkers.User)
                    role = AuthorRole.User;
                else if (input == PromptMarkers.Tools)
                    role = AuthorRole.Unknown;
                else if (input == PromptMarkers.Tool)
                    role = AuthorRole.User;
            }
            else
            {
                if (role == AuthorRole.Unknown && input.Length > 0)
                {
                    toolJsonBuilder.Append(input);
                    toolJsonBuilder.Append('\n');
                }
                else
                    messageBuffer.AppendLine(input);
            }
        }

        flushMessage();

        string rawToolsJson = toolJsonBuilder.ToString();
        ToolDefinition[] tools = [];

        if (toolJsonBuilder.Length > 0)
        {
            try
            {
                tools = ToolDefinition.ParseAll(rawToolsJson.Trim());
            }
            catch (JsonException) { }
        }

        string userInput = "";

        for (int i = chatHistory.Messages.Count - 1; i >= 0; i--)
        {
            if (chatHistory.Messages[i].AuthorRole == AuthorRole.User)
            {
                userInput = chatHistory.Messages[i].Content;
                chatHistory.Messages.RemoveAt(i);
                break;
            }
        }

        return new ParsedPrompt(chatHistory, userInput, tools, rawToolsJson, toolResults.ToArray());
    }

    string BuildSystemPromptWithTools(string existingSystemPrompt, ToolDefinition[] tools, string rawToolsJson)
    {
        if (tools.Length == 0 || ToolHandler.Format == ToolCallFormat.None)
            return existingSystemPrompt;

        if (_cachedSystemPromptWithTools is not null && _cachedToolsJson == rawToolsJson)
            return _cachedSystemPromptWithTools;

        _cachedToolsJson = rawToolsJson;
        _cachedSystemPromptWithTools = ToolHandler.InjectToolSchema(existingSystemPrompt, tools);
        return _cachedSystemPromptWithTools;
    }

    public async Task<string> AskAsync(string rawPrompt)
    {
        var parsed = ParsePrompt(rawPrompt);

        if (parsed.Tools.Length > 0 && ToolHandler.Format != ToolCallFormat.None)
            return await RunWithToolsAsync(parsed);

        return await RunPlainAsync(parsed);
    }

    async Task<string> RunPlainAsync(ParsedPrompt parsed)
    {
        var chatHistory = parsed.ChatHistory;

        for (int i = 0; i < chatHistory.Messages.Count; i++)
        {
            if (chatHistory.Messages[i].AuthorRole == AuthorRole.System)
            {
                chatHistory.Messages[i] = new ChatHistory.Message(
                    AuthorRole.System,
                    BuildSystemPromptWithTools(chatHistory.Messages[i].Content, parsed.Tools, parsed.RawToolsJson));
                break;
            }
        }

        var session = new ChatSession(Executor, chatHistory);

        var inferenceParams = new InferenceParams
        {
            MaxTokens = MaxTokens,
            AntiPrompts = ["User:"]
        };

        var result = new StringBuilder();

        await foreach (var text in session.ChatAsync(
            new ChatHistory.Message(AuthorRole.User, parsed.UserInput),
            inferenceParams))
            result.Append(text);

        return result.ToString();
    }

    async Task<string> RunWithToolsAsync(ParsedPrompt parsed)
    {
        var chatHistory = parsed.ChatHistory;

        for (int i = 0; i < chatHistory.Messages.Count; i++)
        {
            if (chatHistory.Messages[i].AuthorRole == AuthorRole.System)
            {
                chatHistory.Messages[i] = new ChatHistory.Message(
                    AuthorRole.System,
                    BuildSystemPromptWithTools(chatHistory.Messages[i].Content, parsed.Tools, parsed.RawToolsJson));
                break;
            }
        }

        if (parsed.ToolResults.Length > 0)
        {
            var toolResultBlock = new StringBuilder();
            foreach (var (name, result) in parsed.ToolResults)
                toolResultBlock.AppendLine(ToolHandler.FormatToolResult(name, result));

            chatHistory.AddMessage(AuthorRole.User, toolResultBlock.ToString().TrimEnd());
        }

        var session = new ChatSession(Executor, chatHistory);

        var inferenceParams = new InferenceParams
        {
            MaxTokens = MaxTokens,
            AntiPrompts = ["User:"]
        };

        var rawOutput = new StringBuilder();

        await foreach (var text in session.ChatAsync(
            new ChatHistory.Message(AuthorRole.User, parsed.UserInput),
            inferenceParams))
            rawOutput.Append(text);

        string output = rawOutput.ToString();
        var calls = ToolHandler.TryParseToolCalls(output);

        return calls is { Length: > 0 } ? SerializeToolCallsForIpc(calls) : output;
    }

    static string SerializeToolCallsForIpc(DetectedToolCall[] calls)
    {
        var ipcPayload = new System.Text.Json.Nodes.JsonArray();

        foreach (var call in calls)
        {
            System.Text.Json.Nodes.JsonNode? argsNode;
            try
            {
                argsNode = call.Arguments.ValueKind == JsonValueKind.Undefined
                    ? new System.Text.Json.Nodes.JsonObject()
                    : System.Text.Json.Nodes.JsonNode.Parse(call.Arguments.GetRawText());
            }
            catch (JsonException)
            {
                argsNode = new System.Text.Json.Nodes.JsonObject();
            }

            ipcPayload.Add(new System.Text.Json.Nodes.JsonObject
            {
                ["function"] = new System.Text.Json.Nodes.JsonObject
                {
                    ["name"] = call.Name,
                    ["arguments"] = argsNode
                }
            });
        }

        return PromptMarkers.ToolCallsPrefix + ipcPayload.ToJsonString();
    }

    public string Ask(string prompt) => AskAsync(prompt).Result;
}

record ParsedPrompt(
    ChatHistory ChatHistory,
    string UserInput,
    ToolDefinition[] Tools,
    string RawToolsJson,
    (string Name, string Result)[] ToolResults);

static class Program {
    static string WaitForPrompt(string fileName) {
        while (true)
        {
            if (File.Exists(fileName))
            {
                StreamReader promptStream = new StreamReader(fileName);

                string prompt = promptStream.ReadToEnd();

                promptStream.Close();

                File.Delete(fileName);

                return prompt;
            }

            Thread.Sleep(100);
        }
    }

    [STAThread]
    static void Main(string[] args)
    {
        Thread.CurrentThread.CurrentCulture = CultureInfo.CreateSpecificCulture("en-US");

        try
        {
            LLMExecutor executor = new LLMExecutor(args[2],
                                                   (args.Length > 3) ? Double.Parse(args[3]) : 0.5,
                                                   (args.Length > 4) ? int.Parse(args[4]) : 2048,
                                                   (args.Length > 5) ? int.Parse(args[5]) : 0,
                                                   (args.Length > 6) ? args[6] : "");

            while (true)
            {
                string prompt = WaitForPrompt(args[0]);

                if (prompt.Trim() == "Exit")
                    break;

                try
                {
                    string answer = executor.Ask(prompt);
                    StreamWriter outStream = new StreamWriter(args[1], false, Encoding.Unicode);

                    outStream.Write(answer);
                    outStream.Flush();

                    outStream.Close();
                }
                catch (Exception)
                {
                    StreamWriter outStream = new StreamWriter(args[1], false, Encoding.Unicode);

                    outStream.Write("Error");
                    outStream.Flush();

                    outStream.Close();
                }
            }
        }
        catch (Exception)
        {
            System.Environment.Exit(1);
        }
    }
}
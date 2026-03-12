using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace LLMRuntime;

enum ToolCallFormat
{
    None,
    Llama3,
    Mistral,
    GenericJson
}

interface IToolCallHandler
{
    ToolCallFormat Format { get; }
    string InjectToolSchema(string systemPrompt, ToolDefinition[] tools);
    DetectedToolCall[]? TryParseToolCalls(string rawOutput);
    string FormatToolResult(string toolName, string result);
    bool IsCapable(string rawOutput);
}

static class ToolCallHandlerFactory
{
    static readonly Dictionary<ToolCallFormat, IToolCallHandler> Handlers = new()
    {
        [ToolCallFormat.Llama3] = new Llama3ToolCallHandler(),
        [ToolCallFormat.Mistral] = new MistralToolCallHandler(),
        [ToolCallFormat.GenericJson] = new GenericJsonToolCallHandler(),
    };

    public static IToolCallHandler Resolve(string formatName) =>
        formatName.ToLowerInvariant() switch
        {
            "llama3" or "llama-3" or "llama3.1" or "llama3.2" or "llama3.3" => Handlers[ToolCallFormat.Llama3],
            "mistral" => Handlers[ToolCallFormat.Mistral],
            "generic" or "genericjson" or "json" => Handlers[ToolCallFormat.GenericJson],
            _ => Handlers[ToolCallFormat.GenericJson]
        };

    public static IToolCallHandler None { get; } = new NullToolCallHandler();
}

sealed class NullToolCallHandler : IToolCallHandler
{
    public ToolCallFormat Format => ToolCallFormat.None;
    public string InjectToolSchema(string systemPrompt, ToolDefinition[] tools) => systemPrompt;
    public DetectedToolCall[]? TryParseToolCalls(string rawOutput) => null;
    public string FormatToolResult(string toolName, string result) => result;
    public bool IsCapable(string rawOutput) => false;
}

sealed class Llama3ToolCallHandler : IToolCallHandler
{
    const string PythonTag = "<|python_tag|>";
    const string EomTag = "<|eom_id|>";

    public ToolCallFormat Format => ToolCallFormat.Llama3;

    public string InjectToolSchema(string systemPrompt, ToolDefinition[] tools)
    {
        if (tools.Length == 0)
            return systemPrompt;

        var sb = new StringBuilder(systemPrompt.Length + 512);
        sb.Append(systemPrompt);
        sb.AppendLine();
        sb.AppendLine("You have access to the following tools. To call a tool, respond with a JSON object wrapped in <|python_tag|> tags:");
        sb.AppendLine("[");
        for (int i = 0; i < tools.Length; i++)
        {
            sb.Append("  ");
            sb.Append(tools[i].ToOpenAISchema());
            if (i < tools.Length - 1)
                sb.AppendLine(",");
            else
                sb.AppendLine();
        }
        sb.AppendLine("]");
        sb.AppendLine("Call tools by outputting: <|python_tag|>{\"name\": \"tool_name\", \"parameters\": {\"param\": \"value\"}}");
        sb.Append("After receiving tool results, continue your response normally.");
        return sb.ToString();
    }

    public DetectedToolCall[]? TryParseToolCalls(string rawOutput)
    {
        int tagStart = rawOutput.IndexOf(PythonTag, StringComparison.Ordinal);
        if (tagStart < 0)
            return null;

        int jsonStart = tagStart + PythonTag.Length;
        int jsonEnd = rawOutput.IndexOf(EomTag, jsonStart, StringComparison.Ordinal);
        string jsonSlice = jsonEnd >= 0
            ? rawOutput[jsonStart..jsonEnd]
            : rawOutput[jsonStart..];

        jsonSlice = jsonSlice.Trim();
        if (string.IsNullOrEmpty(jsonSlice))
            return null;

        return TryParseToolCallJson(jsonSlice);
    }

    public string FormatToolResult(string toolName, string result) =>
        $"<|start_header_id|>ipython<|end_header_id|>\n{result}<|eot_id|>";

    public bool IsCapable(string rawOutput) =>
        rawOutput.Contains(PythonTag, StringComparison.Ordinal);

    static DetectedToolCall[]? TryParseToolCallJson(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            return ParseOneOrMany(doc.RootElement);
        }
        catch (JsonException)
        {
            return null;
        }
    }

    static DetectedToolCall[]? ParseOneOrMany(JsonElement el)
    {
        if (el.ValueKind == JsonValueKind.Array)
        {
            var calls = new List<DetectedToolCall>();
            foreach (var item in el.EnumerateArray())
            {
                var call = ParseSingle(item);
                if (call is not null)
                    calls.Add(call);
            }
            return calls.Count > 0 ? calls.ToArray() : null;
        }

        if (el.ValueKind == JsonValueKind.Object)
        {
            var single = ParseSingle(el);
            return single is not null ? [single] : null;
        }

        return null;
    }

    static DetectedToolCall? ParseSingle(JsonElement el)
    {
        if (el.ValueKind != JsonValueKind.Object)
            return null;

        string? name = null;
        JsonElement args = default;

        if (el.TryGetProperty("name", out var n))
            name = n.GetString();

        if (el.TryGetProperty("parameters", out var p))
            args = p;
        else if (el.TryGetProperty("arguments", out var a))
            args = a;

        if (string.IsNullOrEmpty(name))
            return null;

        return DetectedToolCall.Create(name, args);
    }
}

sealed class MistralToolCallHandler : IToolCallHandler
{
    const string ToolCallsTag = "[TOOL_CALLS]";
    const string ToolResultsTag = "[TOOL_RESULTS]";

    public ToolCallFormat Format => ToolCallFormat.Mistral;

    public string InjectToolSchema(string systemPrompt, ToolDefinition[] tools)
    {
        if (tools.Length == 0)
            return systemPrompt;

        var sb = new StringBuilder(systemPrompt.Length + 512);
        sb.Append(systemPrompt);
        sb.AppendLine();
        sb.AppendLine("You have access to the following tools:");
        sb.AppendLine("[");
        for (int i = 0; i < tools.Length; i++)
        {
            sb.Append("  ");
            sb.Append(tools[i].ToOpenAISchema());
            if (i < tools.Length - 1)
                sb.AppendLine(",");
            else
                sb.AppendLine();
        }
        sb.AppendLine("]");
        sb.Append("To call tools, output [TOOL_CALLS] followed by a JSON array of calls: [{\"name\":\"tool_name\",\"arguments\":{\"param\":\"value\"}}]");
        return sb.ToString();
    }

    public DetectedToolCall[]? TryParseToolCalls(string rawOutput)
    {
        int tagPos = rawOutput.IndexOf(ToolCallsTag, StringComparison.Ordinal);
        if (tagPos < 0)
            return null;

        int jsonStart = rawOutput.IndexOf('[', tagPos + ToolCallsTag.Length);
        if (jsonStart < 0)
            return null;

        int jsonEnd = FindMatchingBracket(rawOutput, jsonStart, '[', ']');
        if (jsonEnd < 0)
            return null;

        string jsonSlice = rawOutput[jsonStart..(jsonEnd + 1)];

        try
        {
            using var doc = JsonDocument.Parse(jsonSlice);
            if (doc.RootElement.ValueKind != JsonValueKind.Array)
                return null;

            var calls = new List<DetectedToolCall>();
            foreach (var item in doc.RootElement.EnumerateArray())
            {
                if (item.ValueKind != JsonValueKind.Object)
                    continue;

                string? name = item.TryGetProperty("name", out var n) ? n.GetString() : null;
                if (string.IsNullOrEmpty(name))
                    continue;

                JsonElement args = item.TryGetProperty("arguments", out var a) ? a : default;
                calls.Add(DetectedToolCall.Create(name, args));
            }

            return calls.Count > 0 ? calls.ToArray() : null;
        }
        catch (JsonException)
        {
            return null;
        }
    }

    public string FormatToolResult(string toolName, string result) =>
        $"[TOOL_RESULTS]{{\"call\": {{\"name\": \"{toolName}\"}}, \"content\": {JsonSerializer.Serialize(result)}}}[/TOOL_RESULTS]";

    public bool IsCapable(string rawOutput) =>
        rawOutput.Contains(ToolCallsTag, StringComparison.Ordinal);

    static int FindMatchingBracket(string text, int openPos, char open, char close)
    {
        int depth = 0;
        for (int i = openPos; i < text.Length; i++)
        {
            if (text[i] == open) depth++;
            else if (text[i] == close) { depth--; if (depth == 0) return i; }
        }
        return -1;
    }
}

sealed class GenericJsonToolCallHandler : IToolCallHandler
{
    const string ToolCallMarker = "\"tool_calls\"";
    const string FunctionCallAlt = "\"function_call\"";

    public ToolCallFormat Format => ToolCallFormat.GenericJson;

    public string InjectToolSchema(string systemPrompt, ToolDefinition[] tools)
    {
        if (tools.Length == 0)
            return systemPrompt;

        var sb = new StringBuilder(systemPrompt.Length + 512);
        sb.Append(systemPrompt);
        sb.AppendLine();
        sb.AppendLine("You have access to the following tools. To use a tool, respond ONLY with a JSON object in this exact format:");
        sb.AppendLine("{\"tool_calls\": [{\"function\": {\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}}]}");
        sb.AppendLine("Available tools:");
        sb.AppendLine("[");
        for (int i = 0; i < tools.Length; i++)
        {
            sb.Append("  ");
            sb.Append(tools[i].ToOpenAISchema());
            if (i < tools.Length - 1)
                sb.AppendLine(",");
            else
                sb.AppendLine();
        }
        sb.AppendLine("]");
        sb.Append("When not calling a tool, respond normally in plain text.");
        return sb.ToString();
    }

    public DetectedToolCall[]? TryParseToolCalls(string rawOutput)
    {
        string trimmed = rawOutput.Trim();

        int objStart = trimmed.IndexOf('{');
        if (objStart < 0)
            return null;

        int objEnd = FindMatchingBracket(trimmed, objStart, '{', '}');
        if (objEnd < 0)
            return null;

        string candidate = trimmed[objStart..(objEnd + 1)];

        if (!candidate.Contains(ToolCallMarker, StringComparison.Ordinal) &&
            !candidate.Contains(FunctionCallAlt, StringComparison.Ordinal))
            return null;

        try
        {
            using var doc = JsonDocument.Parse(candidate);
            var root = doc.RootElement;

            if (root.TryGetProperty("tool_calls", out var toolCalls) &&
                toolCalls.ValueKind == JsonValueKind.Array)
                return ParseToolCallArray(toolCalls);

            if (root.TryGetProperty("function_call", out var funcCall) &&
                funcCall.ValueKind == JsonValueKind.Object)
            {
                string? name = funcCall.TryGetProperty("name", out var n) ? n.GetString() : null;
                if (string.IsNullOrEmpty(name))
                    return null;

                JsonElement args = funcCall.TryGetProperty("arguments", out var a) ? a : default;
                return [new DetectedToolCall(name, args)];
            }

            return null;
        }
        catch (JsonException)
        {
            return null;
        }
    }

    public string FormatToolResult(string toolName, string result) =>
        $"Tool result for {toolName}: {result}";

    public bool IsCapable(string rawOutput) =>
        rawOutput.Contains(ToolCallMarker, StringComparison.Ordinal) ||
        rawOutput.Contains(FunctionCallAlt, StringComparison.Ordinal);

    static DetectedToolCall[]? ParseToolCallArray(JsonElement arr)
    {
        var calls = new List<DetectedToolCall>();

        foreach (var item in arr.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Object)
                continue;

            JsonElement fn = default;
            if (!item.TryGetProperty("function", out fn) || fn.ValueKind != JsonValueKind.Object)
            {
                if (item.TryGetProperty("name", out var directName))
                {
                    string? name = directName.GetString();
                    if (string.IsNullOrEmpty(name))
                        continue;
                    JsonElement args = item.TryGetProperty("arguments", out var a) ? a : default;
                    calls.Add(new DetectedToolCall(name, args));
                    continue;
                }
                continue;
            }

            string? fnName = fn.TryGetProperty("name", out var nn) ? nn.GetString() : null;
            if (string.IsNullOrEmpty(fnName))
                continue;

            JsonElement fnArgs = fn.TryGetProperty("arguments", out var fa) ? fa : default;
            if (fnArgs.ValueKind == JsonValueKind.String)
            {
                string? argsStr = fnArgs.GetString();
                if (!string.IsNullOrEmpty(argsStr))
                {
                    try
                    {
                        using var argDoc = JsonDocument.Parse(argsStr);
                        fnArgs = argDoc.RootElement.Clone();
                    }
                    catch (JsonException) { }
                }
            }

            calls.Add(new DetectedToolCall(fnName, fnArgs));
        }

        return calls.Count > 0 ? calls.ToArray() : null;
    }

    static int FindMatchingBracket(string text, int openPos, char open, char close)
    {
        int depth = 0;
        bool inString = false;
        bool escaped = false;

        for (int i = openPos; i < text.Length; i++)
        {
            char c = text[i];

            if (escaped) { escaped = false; continue; }
            if (c == '\\' && inString) { escaped = true; continue; }
            if (c == '"') { inString = !inString; continue; }
            if (inString) continue;

            if (c == open) depth++;
            else if (c == close) { depth--; if (depth == 0) return i; }
        }

        return -1;
    }
}

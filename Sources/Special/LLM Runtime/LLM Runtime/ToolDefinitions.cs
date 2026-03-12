using System.Text.Json;
using System.Text.Json.Nodes;

namespace LLMRuntime;

record ToolParameter(string Name, string Description, string Type, string[]? Enum, bool Required);

record ToolDefinition(string Name, string Description, ToolParameter[] Parameters)
{
    public static ToolDefinition[] ParseAll(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        if (root.ValueKind != JsonValueKind.Array)
            return [];

        var tools = new List<ToolDefinition>();

        foreach (var element in root.EnumerateArray())
        {
            if (!element.TryGetProperty("type", out var typeEl) || typeEl.GetString() != "function")
                continue;

            if (!element.TryGetProperty("function", out var fn))
                continue;

            string name = fn.TryGetProperty("name", out var n) ? n.GetString() ?? "" : "";
            string description = fn.TryGetProperty("description", out var d) ? d.GetString() ?? "" : "";

            if (string.IsNullOrEmpty(name))
                continue;

            var parameters = new List<ToolParameter>();

            if (fn.TryGetProperty("parameters", out var paramsEl))
            {
                var required = new HashSet<string>(StringComparer.Ordinal);

                if (paramsEl.TryGetProperty("required", out var reqEl))
                    foreach (var r in reqEl.EnumerateArray())
                        if (r.GetString() is { } rName)
                            required.Add(rName);

                if (paramsEl.TryGetProperty("properties", out var propsEl))
                {
                    foreach (var prop in propsEl.EnumerateObject())
                    {
                        string paramType = prop.Value.TryGetProperty("type", out var pt) ? pt.GetString() ?? "string" : "string";
                        string paramDesc = prop.Value.TryGetProperty("description", out var pd) ? pd.GetString() ?? "" : "";

                        string[]? enumValues = null;
                        if (prop.Value.TryGetProperty("enum", out var enumEl))
                        {
                            var ev = new List<string>();
                            foreach (var e in enumEl.EnumerateArray())
                                if (e.GetString() is { } s)
                                    ev.Add(s);
                            enumValues = ev.Count > 0 ? ev.ToArray() : null;
                        }

                        parameters.Add(new ToolParameter(prop.Name, paramDesc, paramType, enumValues, required.Contains(prop.Name)));
                    }
                }
            }

            tools.Add(new ToolDefinition(name, description, parameters.ToArray()));
        }

        return tools.ToArray();
    }

    public string ToOpenAISchema()
    {
        var properties = new JsonObject();
        var requiredList = new JsonArray();

        foreach (var p in Parameters)
        {
            var propObj = new JsonObject
            {
                ["type"] = p.Type,
                ["description"] = p.Description
            };

            if (p.Enum is { Length: > 0 })
            {
                var enumArr = new JsonArray();
                foreach (var e in p.Enum)
                    enumArr.Add(e);
                propObj["enum"] = enumArr;
            }

            properties[p.Name] = propObj;

            if (p.Required)
                requiredList.Add(p.Name);
        }

        var schema = new JsonObject
        {
            ["type"] = "function",
            ["function"] = new JsonObject
            {
                ["name"] = Name,
                ["description"] = Description,
                ["parameters"] = new JsonObject
                {
                    ["type"] = "object",
                    ["properties"] = properties,
                    ["required"] = requiredList
                }
            }
        };

        return schema.ToJsonString();
    }
}

record DetectedToolCall(string Name, JsonElement Arguments)
{
    public static DetectedToolCall Create(string name, JsonElement arguments) =>
        new(name, arguments.ValueKind == JsonValueKind.Undefined ? default : arguments.Clone());
}

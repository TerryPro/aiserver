from aiserver.ju_analysis import get_library_metadata

library = get_library_metadata()

print("Library Categories:", list(library.keys()))

# Check a specific algorithm
category = "数据预处理"
if category in library:
    print(f"\nChecking category: {category}")
    funcs = library[category]
    if funcs:
        first_func = funcs[0]
        print(f"First function: {first_func['name']} (ID: {first_func['id']})")
        print("Template present:", "template" in first_func)
        if "template" in first_func:
            print("Template preview (first 50 chars):", first_func['template'][:50].replace('\n', '\\n'))
    else:
        print("No functions in this category.")
else:
    print(f"Category {category} not found.")

# Check trend plot category
category = "趋势绘制"
if category in library:
    print(f"\nChecking category: {category}")
    funcs = library[category]
    print(f"Found {len(funcs)} algorithms.")
    for f in funcs:
        print(f"- {f['name']}: Template length {len(f.get('template', ''))}")

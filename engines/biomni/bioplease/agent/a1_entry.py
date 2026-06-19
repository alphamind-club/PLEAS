if __name__ == "__main__":
    # Example CLI usage: python a1.py "your prompt here" -fastmode
    prompt = None
    fastmode = parse_fastmode_from_argv()
    for arg in sys.argv[1:]:
        if not arg.startswith('-'):
            prompt = arg
            break
    if prompt:
        agent = A1(fastmode=fastmode)
        agent.go(prompt)
    else:
        print("Usage: python a1.py 'your prompt here' [-fastmode]")

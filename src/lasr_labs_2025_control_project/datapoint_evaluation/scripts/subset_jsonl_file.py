import json
import sys


def split_jsonl(input_file: str, output_file: str, n_attacks: int, m_honeypots: int):
    """
    Extract first n attack samples (YES answers) and m honeypot samples (NO answers)
    from a JSONL file and write to a new JSONL file.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        n_attacks: Number of attack samples to extract
        m_honeypots: Number of honeypot samples to extract
    """
    attacks = []
    honeypots = []

    # Read the input file and categorize samples
    with open(input_file, "r") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            # Check if this is a conversational format with messages
            if isinstance(data, dict) and "messages" in data:
                messages = data["messages"]
            elif isinstance(data, list):
                messages = data
            else:
                # Single message format - check the content directly
                messages = [data]

            # Look for assistant's answer
            is_attack = False
            is_honeypot = False

            for msg in messages:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if "<answer>YES</answer>" in content:
                        is_attack = True
                        break
                    elif "<answer>NO</answer>" in content:
                        is_honeypot = True
                        break

            # Add to appropriate category if we haven't reached the limit
            if is_attack and len(attacks) < n_attacks:
                attacks.append(line)
            elif is_honeypot and len(honeypots) < m_honeypots:
                honeypots.append(line)

            # Stop early if we've collected enough samples
            if len(attacks) >= n_attacks and len(honeypots) >= m_honeypots:
                break

    # Write output file
    with open(output_file, "w") as f:
        for line in attacks:
            f.write(line)
        for line in honeypots:
            f.write(line)

    print("Extraction complete!")
    print(f"Attacks collected: {len(attacks)}/{n_attacks}")
    print(f"Honeypots collected: {len(honeypots)}/{m_honeypots}")
    print(f"Total samples: {len(attacks) + len(honeypots)}")
    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python split_jsonl.py <input_file> <output_file>")
        print("Example: python split_jsonl.py data.jsonl output.jsonl")
        print("Note: Extracts 26 attacks and 36 honeypots")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    n_attacks = 52
    m_honeypots = 72

    split_jsonl(input_file, output_file, n_attacks, m_honeypots)

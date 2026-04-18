#!/usr/bin/env python3
"""Generate phonetically-diverse validation prompts per language.

Designed for native-speaker grading. Each language gets sentences that
cover:
    - greetings and high-frequency words
    - prenasalised stops (mb/nd/ng/nz)
    - aspirated stops (ph/th/kh)
    - language-specific phonemes (whistled fricatives for Shona; clicks for
      Nguni)
    - simple numerals 0-10
    - one code-switched English insertion

The output is JSONL at ``data/validation/<language>.jsonl`` with fields
``id``, ``text``, ``focus``. The synthesis script reads these and produces
audio for grading.
"""

import argparse
import json
from pathlib import Path


PROMPTS: dict[str, list[dict[str, str]]] = {
    "shona": [
        {"id": "sho_greet_01", "focus": "greeting", "text": "Mhoroi, makadii nhasi"},
        {"id": "sho_pren_01", "focus": "prenasalized", "text": "Mbira inorira"},
        {"id": "sho_pren_02", "focus": "prenasalized", "text": "Ndinoda mvura"},
        {"id": "sho_whistle_01", "focus": "whistled fricative", "text": "Zvakanaka chaizvo"},
        {"id": "sho_asp_01", "focus": "aspirated", "text": "Ndinopha pamba"},
        {"id": "sho_num_01", "focus": "numerals", "text": "Ndine mbiri nemoto"},
        {"id": "sho_codesw_01", "focus": "code-switching", "text": "Ndiri busy nhasi"},
    ],
    "ndebele": [
        {"id": "nde_greet_01", "focus": "greeting", "text": "Sawubona, unjani lamuhla"},
        {"id": "nde_click_01", "focus": "click", "text": "Iqanda lihle"},
        {"id": "nde_pren_01", "focus": "prenasalized", "text": "Ngiyabonga kakhulu"},
        {"id": "nde_asp_01", "focus": "aspirated", "text": "Ukhuluma kuhle"},
        {"id": "nde_num_01", "focus": "numerals", "text": "Ngifuna kunye"},
    ],
    "zulu": [
        {"id": "zul_greet_01", "focus": "greeting", "text": "Sawubona, unjani namuhla"},
        {"id": "zul_click_dental", "focus": "click dental", "text": "Icala likhulu"},
        {"id": "zul_click_alv", "focus": "click alveolar", "text": "Iqanda elisha"},
        {"id": "zul_click_lat", "focus": "click lateral", "text": "Ixoxo elidala"},
        {"id": "zul_click_nasal", "focus": "click nasal", "text": "Ngqondo enhle"},
        {"id": "zul_pren_01", "focus": "prenasalized", "text": "Ngiyabonga kakhulu"},
        {"id": "zul_codesw_01", "focus": "code-switching", "text": "Ngiyakwenza ishopping namuhla"},
    ],
    "xhosa": [
        {"id": "xho_greet_01", "focus": "greeting", "text": "Molo, unjani namhlanje"},
        {"id": "xho_click_dental", "focus": "click dental", "text": "Icici elihle"},
        {"id": "xho_click_alv", "focus": "click alveolar", "text": "Iqaqa lihle"},
        {"id": "xho_click_asp", "focus": "click aspirated", "text": "Xhosa lulwimi olukhulu"},
        {"id": "xho_pren_01", "focus": "prenasalized", "text": "Enkosi kakhulu"},
        # TODO(native-speaker): authentic Xhosa code-switching example.
        # Zulu uses prefix-fused loanwords (ishopping, eshopping); confirm the
        # equivalent Xhosa pattern (ndiyenza ishopping? ndiya kwi-office?)
        # before adding here. Do NOT invent hyphenated forms.
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/validation", help="Where to write per-language JSONL files")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for language, prompts in PROMPTS.items():
        path = out_dir / f"{language}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for prompt in prompts:
                fh.write(json.dumps(prompt, ensure_ascii=False) + "\n")
        print(f"wrote {len(prompts)} prompts to {path}")


if __name__ == "__main__":
    main()

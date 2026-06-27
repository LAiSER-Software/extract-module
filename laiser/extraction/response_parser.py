import json
import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class ResponseParser:
    """Parse model responses into structured extractor outputs."""

    @staticmethod
    def _parse_skills_from_response(response: str) -> List[str]:
        if not response or not response.strip():
            return []

        fragments: List[str] = []
        stripped = response.strip()

        code_match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
        if code_match:
            fragments.append(code_match.group(1).strip())

        brace_match = re.search(r"\{.*?\}", stripped, re.DOTALL)
        if brace_match:
            fragments.append(brace_match.group(0).strip())

        list_match = re.search(r"\[.*?\]", stripped, re.DOTALL)
        if list_match:
            fragments.append(list_match.group(0).strip())

        fragments.append(stripped)

        seen = set()
        for fragment in fragments:
            if not fragment or fragment in seen:
                continue
            seen.add(fragment)
            try:
                loaded = json.loads(fragment)
            except json.JSONDecodeError:
                continue

            if isinstance(loaded, dict):
                skills = loaded.get("skills")
                if isinstance(skills, list):
                    return [str(s).strip() for s in skills if str(s).strip()]
            elif isinstance(loaded, list):
                return [str(s).strip() for s in loaded if str(s).strip()]

        quoted_skills = re.findall(r"\"([^\"]{1,100})\"", stripped)
        if quoted_skills:
            cleaned = []
            for skill in quoted_skills:
                skill = skill.strip()
                if not skill:
                    continue
                if not (1 <= len(skill.split()) <= 5):
                    continue
                if skill.lower().startswith("skills"):
                    continue
                cleaned.append(skill)
            if cleaned:
                return cleaned

        return []

    @staticmethod
    def parse_skill_extraction_response(response: str) -> List[str]:
        try:
            if not response:
                return []

            pattern = r"<start_of_turn>model\\s*<eos>(.*?)<eos>\\s*$"
            match = re.search(pattern, response, re.DOTALL)

            if match:
                content = match.group(1).strip()
                lines = [line.strip() for line in content.split("\\n") if line.strip()]
                return [line[1:].strip() for line in lines if line.startswith("-")]

            lines = [line.strip() for line in response.split("\n") if line.strip()]
            clean_lines = []
            for line in lines:
                if line.startswith("<start_of_turn>") or line.startswith("<end_of_turn>"):
                    continue
                if "--" in line:
                    continue
                clean_lines.append(line)

            return clean_lines
        except Exception as exc:
            print(f"Warning: Failed to parse skill extraction response: {exc}")
            return []

    @staticmethod
    def parse_ksa_extraction_response(response: str) -> List[Dict[str, Any]]:
        try:
            if not response:
                return []

            out = []
            items = [item.strip() for item in response.split("->") if item.strip()]
            for i, item in enumerate(items):
                skill_data = {}
                try:
                    skill_match = re.search(r"Skill:\s*([^,\n]+)", item)
                    if skill_match:
                        skill_data["Skill"] = skill_match.group(1).strip()

                    level_match = re.search(r"Level:\s*(\d+)", item)
                    if level_match:
                        skill_data["Level"] = int(level_match.group(1).strip())

                    knowledge_match = re.search(
                        r"Knowledge Required:\s*(.*?)(?=\s*Task Abilities:|\s*$)",
                        item,
                        re.DOTALL,
                    )
                    if knowledge_match:
                        knowledge_raw = knowledge_match.group(1).strip()
                        skill_data["Knowledge Required"] = [k.strip() for k in knowledge_raw.split(",") if k.strip()]

                    task_match = re.search(r"Task Abilities:\s*(.*?)(?=\s*$)", item, re.DOTALL)
                    if task_match:
                        task_raw = task_match.group(1).strip()
                        skill_data["Task Abilities"] = [t.strip() for t in task_raw.split(",") if t.strip()]

                    if skill_data:
                        out.append(skill_data)
                except Exception as exc:
                    print(f"Warning: Error processing KSA item {i}: {exc}")
                    continue

            return out
        except Exception as exc:
            print(f"Warning: Failed to parse KSA extraction response: {exc}")
            return []

    @staticmethod
    def parse_knowledge_task_response(response: str) -> List[Dict[str, Any]]:
        if not response or not response.strip():
            return []

        def validate(results: Any) -> List[Dict[str, Any]]:
            validated = []
            if results is None:
                return validated
            if not isinstance(results, list):
                results = [results]

            for item in results:
                if not isinstance(item, dict):
                    continue
                skill = str(item.get("skill", item.get("Skill", ""))).strip()
                knowledge = item.get("knowledge", item.get("Knowledge", item.get("Knowledge Required", [])))
                tasks = item.get("tasks", item.get("task", item.get("Task Abilities", [])))

                if not isinstance(knowledge, list):
                    knowledge = [str(knowledge)] if knowledge else []
                if not isinstance(tasks, list):
                    tasks = [str(tasks)] if tasks else []

                if skill:
                    validated.append(
                        {
                            "skill": skill,
                            "knowledge": [str(k).strip() for k in knowledge if str(k).strip()],
                            "tasks": [str(t).strip() for t in tasks if str(t).strip()],
                        }
                    )
            return validated

        def parse_array_fragment(fragment: str) -> List[str]:
            fragment = fragment.strip()
            if not fragment:
                return []
            try:
                parsed = json.loads(fragment)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
            return [m.strip() for m in re.findall(r'"([^"]+)"', fragment) if m.strip()]

        try:
            candidates: List[str] = []
            stripped = response.strip()

            fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
            if fenced:
                candidates.append(fenced.group(1).strip())

            obj_match = re.search(r"\{.*\}", stripped, re.DOTALL)
            if obj_match:
                candidates.append(obj_match.group(0).strip())

            list_match = re.search(r"\[.*\]", stripped, re.DOTALL)
            if list_match:
                candidates.append(list_match.group(0).strip())

            candidates.append(stripped)

            seen = set()
            for candidate in candidates:
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    continue

                if isinstance(parsed, dict):
                    for key in ("results", "data", "items"):
                        validated = validate(parsed.get(key, []))
                        if validated:
                            return validated

                    validated = validate([parsed])
                    if validated:
                        return validated
                elif isinstance(parsed, list):
                    validated = validate(parsed)
                    if validated:
                        return validated

            block_pattern = re.compile(
                r'"skill"\s*:\s*"(?P<skill>[^"]+)"'
                r'.*?"knowledge"\s*:\s*(?P<knowledge>\[[^\]]*\])'
                r'.*?"tasks"\s*:\s*(?P<tasks>\[[^\]]*\])',
                re.DOTALL | re.IGNORECASE,
            )

            fallback_results: List[Dict[str, Any]] = []
            for match in block_pattern.finditer(stripped):
                skill = match.group("skill").strip()
                knowledge = parse_array_fragment(match.group("knowledge"))
                tasks = parse_array_fragment(match.group("tasks"))
                if skill:
                    fallback_results.append({"skill": skill, "knowledge": knowledge, "tasks": tasks})

            if fallback_results:
                return fallback_results

            logger.warning(
                "Failed to parse knowledge/task response into usable blocks. Preview: %s",
                stripped.replace("\n", " ")[:300],
            )
            return []
        except Exception as exc:
            logger.warning(f"Failed to parse knowledge/task response: {exc}")
            return []

    def parse_ksa_details_response(response: str) -> Tuple[List[str], List[str]]:
        try:
            if not response:
                return [], []

            json_match = re.search(r"\\{.*\\}", response, re.DOTALL)
            if not json_match:
                return [], []

            parsed = json.loads(json_match.group())
            knowledge = parsed.get("Knowledge Required", [])
            task_abilities = parsed.get("Task Abilities", [])

            if not isinstance(knowledge, list):
                knowledge = [str(knowledge)] if knowledge else []
            if not isinstance(task_abilities, list):
                task_abilities = [str(task_abilities)] if task_abilities else []

            return knowledge, task_abilities
        except Exception as exc:
            print(f"Warning: Failed to parse KSA details response: {exc}")
            return [], []

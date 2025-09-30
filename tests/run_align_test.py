# run_align_test.py
from laiser.services import SkillAlignmentService, DataAccessLayer, FAISSIndexManager

def main():
    raw_skills = [
        "Curriculum development",
        "Technical training",
        "Medicaid state websites",
        "State methodologies",
        "QNXT system",
        "Platform skills",
        "Claim processing",
        "Provider data services",
        "Adult learning principles",
    ]

    data_access = DataAccessLayer()
    faiss_manager = FAISSIndexManager(data_access)
    aligner = SkillAlignmentService(data_access, faiss_manager)

    aligned = aligner.align_skills_to_taxonomy(
        raw_skills=raw_skills,
        document_id="aetna_trainer_001",
        description="POSITION SUMMARY...",
        similarity_threshold=0.30,
        top_k=10
    )

    print("\nAligned Skills:")
    if not aligned:
        print("[No skills aligned]")
    elif isinstance(aligned, list):
        for skill in aligned:
            print(f" - {skill}")
    else:
        print(aligned)

if __name__ == "__main__":
    main()

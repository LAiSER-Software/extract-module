import pandas as pd
from typing import Dict, Any, Tuple

def eda_on_results(
    results: pd.DataFrame,
    *,
    print_report: bool = True
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Perform EDA on an extract-and-align `results` DataFrame.

    Returns (summary_dict, agg_df) where:
      - summary_dict contains numeric summaries and counts
      - agg_df is a grouped table: taxonomy -> count, mean/std/min/max corr

    The function is flexible about column names. It will look for:
      - research id: 'Research ID' (fallback: first column)
      - raw skill: 'Raw Skill' or 'Raw' or 'raw_skill'
      - taxonomy skill: 'Taxonomy Skill' (used for duplicate detection)
      - taxonomy label column: 'Taxonomy Source' or 'taxonomy' or 'Taxonomy Source'
      - correlation: 'Correlation Coefficient' or 'correlation' or 'Similarity'
    """
    if results is None or results.empty:
        if print_report:
            print("EDA: empty results DataFrame")
        return {}, pd.DataFrame()

    df = results.copy()

    # --- column detection / canonical names ---
    def _pick(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    research_col = _pick(["Research ID", "research id", "research_id"]) or df.columns[0]
    raw_col = _pick(["Raw Skill", "raw_skill", "raw skill", "Raw"]) or _pick([df.columns[1] if df.shape[1] > 1 else None])
    tax_skill_col = _pick(["Taxonomy Skill", "taxonomy skill", "taxonomy_skill"])
    taxonomy_col = _pick(["Taxonomy Source", "taxonomy", "Taxonomy Source", "taxonomy_source", "source"])
    corr_col = _pick(["Correlation Coefficient", "correlation coefficient", "Correlation", "correlation", "Similarity", "similarity"])

    # enforce names
    col_map = {}
    if research_col != "Research ID":
        col_map[research_col] = "Research ID"
    if raw_col and raw_col != "Raw Skill":
        col_map[raw_col] = "Raw Skill"
    if tax_skill_col and tax_skill_col != "Taxonomy Skill":
        col_map[tax_skill_col] = "Taxonomy Skill"
    if taxonomy_col and taxonomy_col != "taxonomy":
        col_map[taxonomy_col] = "taxonomy"
    if corr_col and corr_col != "Correlation Coefficient":
        col_map[corr_col] = "Correlation Coefficient"
    if col_map:
        df = df.rename(columns=col_map)

    # Make sure correlation is numeric
    if "Correlation Coefficient" in df.columns:
        df["Correlation Coefficient"] = pd.to_numeric(df["Correlation Coefficient"], errors="coerce")

    # --- core view (small table) ---
    core_cols = ["Research ID", "Raw Skill"]
    if "Taxonomy Skill" in df.columns:
        core_cols.append("Taxonomy Skill")
    if "taxonomy" in df.columns:
        core_cols.append("taxonomy")
    if "Taxonomy Description" in df.columns:
        core_cols.append("Taxonomy Description")
    if "Correlation Coefficient" in df.columns:
        core_cols.append("Correlation Coefficient")

    core_view = df.loc[:, [c for c in core_cols if c in df.columns]]

    # --- taxonomy summary ---
    taxonomy_counts = df["taxonomy"].value_counts() if "taxonomy" in df.columns else pd.Series(dtype=int)

    # --- correlation stats ---
    corr_series = df["Correlation Coefficient"] if "Correlation Coefficient" in df.columns else pd.Series(dtype=float)
    corr_stats = {}
    if not corr_series.dropna().empty:
        corr_stats = {
            "count": int(corr_series.count()),
            "min": float(corr_series.min()),
            "25%": float(corr_series.quantile(0.25)),
            "median": float(corr_series.median()),
            "mean": float(corr_series.mean()),
            "75%": float(corr_series.quantile(0.75)),
            "max": float(corr_series.max()),
            "std": float(corr_series.std())
        }

    # --- top / bottom matches ---
    top_matches = df.sort_values(by="Correlation Coefficient", ascending=False).head(10) if "Correlation Coefficient" in df.columns else pd.DataFrame()
    bottom_matches = df.sort_values(by="Correlation Coefficient", ascending=True).head(10) if "Correlation Coefficient" in df.columns else pd.DataFrame()

    # --- buckets (if correlation available) ---
    bucket_counts = None
    if "Correlation Coefficient" in df.columns:
        bins = [0.0, 0.4, 0.5, 0.6, 0.7, 1.0]
        labels = ["<0.4", "0.4–0.5", "0.5–0.6", "0.6–0.7", "0.7+"]
        df["Corr Bucket"] = pd.cut(df["Correlation Coefficient"], bins=bins, labels=labels, include_lowest=True)
        bucket_counts = df["Corr Bucket"].value_counts().reindex(labels).fillna(0).astype(int)

    # --- duplicate taxonomy mapping (taxonomy skill duplicated) ---
    dup_counts = None
    if "Taxonomy Skill" in df.columns:
        dup = df["Taxonomy Skill"].value_counts()
        dup_counts = dup[dup > 1].sort_values(ascending=False)

    # --- coverage per document ---
    coverage = df.groupby("Research ID")["Raw Skill"].count()

    # --- grouped agg per taxonomy (mean/std/count) ---
    agg_df = pd.DataFrame()
    if "taxonomy" in df.columns and "Correlation Coefficient" in df.columns:
        agg_df = df.groupby("taxonomy")["Correlation Coefficient"].agg(["count", "mean", "std", "min", "median", "max"]).reset_index()
        agg_df = agg_df.sort_values("count", ascending=False)

    # --- print report ---
    if print_report:
        print("\n" + "=" * 80)
        print("EDA REPORT")
        print("=" * 80)
        print("\n-- Core mapping (sample) --")
        print(core_view.to_string(index=False))
        print("\n-- Unique taxonomies (counts) --")
        if taxonomy_counts is not None and not taxonomy_counts.empty:
            print(taxonomy_counts.to_string())
        else:
            print("No taxonomy column found.")

        print("\n-- Correlation stats --")
        if corr_stats:
            for k, v in corr_stats.items():
                print(f"{k}: {v}")
        else:
            print("No correlation column found or it's empty.")

        if not top_matches.empty:
            print("\n-- Top matches --")
            cols_show = [c for c in ["Research ID", "Raw Skill", "Taxonomy Skill", "taxonomy", "Correlation Coefficient"] if c in top_matches.columns]
            print(top_matches[cols_show].to_string(index=False))

        if not bottom_matches.empty:
            print("\n-- Bottom matches --")
            cols_show = [c for c in ["Research ID", "Raw Skill", "Taxonomy Skill", "taxonomy", "Correlation Coefficient"] if c in bottom_matches.columns]
            print(bottom_matches[cols_show].to_string(index=False))

        if bucket_counts is not None:
            print("\n-- Correlation buckets --")
            print(bucket_counts.to_string())

        if dup_counts is not None and not dup_counts.empty:
            print("\n-- Duplicate taxonomy skill hits (counts > 1) --")
            print(dup_counts.to_string())
        else:
            print("\n-- No duplicate taxonomy skill mappings detected --")

        print("\n-- Skills per document --")
        print(coverage.to_string())

        print("\n" + "=" * 80)
        print("END EDA")
        print("=" * 80)

    # --- summary dict for programmatic use ---
    summary = {
        "n_rows": int(df.shape[0]),
        "n_unique_research_ids": int(df["Research ID"].nunique()) if "Research ID" in df.columns else None,
        "n_unique_taxonomies": int(df["taxonomy"].nunique()) if "taxonomy" in df.columns else 0,
        "taxonomy_counts": taxonomy_counts.to_dict() if taxonomy_counts is not None else {},
        "corr_stats": corr_stats,
        "top_matches": top_matches,
        "bottom_matches": bottom_matches,
        "bucket_counts": bucket_counts if bucket_counts is not None else pd.Series(dtype=int),
        "duplicate_taxonomy_skills": dup_counts if dup_counts is not None else pd.Series(dtype=int),
        "coverage_per_doc": coverage
    }

    return summary, agg_df


def sample_data():
    data = pd.DataFrame([
        {
            "Research ID": "aetna_trainer_001",
            "description": (
                "POSITION SUMMARY: This position requires curriculum development, claim processing, "
                "and provider data services experience.\n\n"
                "RESPONSIBILITIES:\n"
                "- Design and deliver training modules for new hires and existing staff.\n"
                "- Collaborate with subject matter experts to create engaging learning materials.\n"
                "- Review claims workflows and develop simulations for hands-on practice.\n"
                "- Analyze provider data services to identify areas for process improvement.\n\n"
                "QUALIFICATIONS:\n"
                "- Bachelor's degree in Healthcare Administration, Education, or related field.\n"
                "- 3+ years of experience in claims processing and curriculum development.\n"
                "- Excellent communication and analytical skills."
            )
        },
        {
            "Research ID": "aetna_trainer_002",
            "description": (
                "Looking for someone with a strong technical training background and knowledge of "
                "Medicaid state websites.\n\n"
                "RESPONSIBILITIES:\n"
                "- Conduct training sessions for teams navigating state Medicaid portals.\n"
                "- Develop technical documentation and job aids for internal users.\n"
                "- Troubleshoot common issues users face when accessing Medicaid websites.\n"
                "- Coordinate with IT teams to ensure training content is up-to-date with policy changes.\n\n"
                "QUALIFICATIONS:\n"
                "- Experience in healthcare IT systems or state Medicaid platforms.\n"
                "- Prior technical training or instructional design experience preferred.\n"
                "- Strong problem-solving skills and ability to work with cross-functional teams."
            )
        },
        {
        "Research ID": "job_001",
        "description": ("""
        Data Scientist About the job Locations: Boston | Chicago | Washington | Pittsburgh | New York | Brooklyn | Manhattan Beach | Dallas | Miami | San Francisco | Seattle | Los Angeles Who We Are Boston Consulting Group partners with leaders in business and society to tackle their most important challenges and capture their greatest opportunities. BCG was the pioneer in business strategy when it was founded in 1963. Today, we help clients with total transformation-inspiring complex change, enabling organizations to grow, building competitive advantage, and driving bottom-line impact. To succeed, organizations must blend digital and human capabilities. Our diverse, global teams bring deep industry and functional expertise and a range of perspectives to spark change. BCG delivers solutions through leading-edge management consulting along with technology and design, corporate and digital venturesâ€”and business purpose. We work in a uniquely collaborative model across the firm and throughout all levels of the client organization, generating results that allow our clients to thrive. We Are BCG X We're a diverse team of more than 3,000 tech experts united by a drive to make a difference. Working across industries and disciplines, we combine our experience and expertise to tackle the biggest challenges faced by society today. We go beyond what was once thought possible, creating new and innovative solutions to the world's most complex problems. Leveraging BCG's global network and partnerships with leading organizations, BCG X provides a stable ecosystem for talent to build game-changing businesses, products, and services from the ground up, all while growing their career. Together, we strive to create solutions that will positively impact the lives of millions. What You'll Do Our BCG X teams own the full analytics value-chain end to end: framing new business challenges, designing innovative algorithms, implementing, and deploying scalable solutions, and enabling colleagues and clients to fully embrace AI. Our product offerings span from fully custom-builds to industry specific leading edge AI software solutions. As a Data Scientist, you'll be part of our rapidly growing team. You'll have the chance to apply data science methods and analytics to real-world business situations across a variety of industries to drive significant business impact. You'll have the chance to partner with clients in a variety of BCG regions and industries, and on key topics like climate change, enabling them to design, build, and deploy new and innovative solutions. Additional responsibilities will include developing and delivering thought leadership in scientific communities and papers as well as leading conferences on behalf of BCG X. Successful candidates are intellectually curious builders who are biased toward action, scrappy, and communicative. We Are Looking For Talented Individuals With a Passion For Data Science, Statistics, Operations Research And Transforming Organizations Into AI Led Innovative Companies. Successful Candidates Possess The Following Comfortable in a client-facing role with the ambition to lead teams Likes to distill complex results or processes into simple, clear visualizations Explain sophisticated data science concepts in an understandable manner Love building things and are comfortable working with modern development tools and writing code collaboratively (bonus points if you have a software development or DevOps experience) Significant experience applying advanced analytics to a variety of business situations and a proven ability to synthesize complex data Deep understanding of modern machine learning techniques and their mathematical underpinnings, and can translate this into business implications for our clients Have strong project management skills What You'll Bring This position is open to students currently pursuing a full-time Bachelors or Masters degree and graduating between December 2025 - August 2026. The deadline to apply for this position is September 16, 2025 at 11:59pm ET. Technologies Programming Languages: Python Additional info You must live within a reasonable commuting distance of your home office. As a member of that office, it is expected you will be in the office as directed. This role puts you on an accelerated path of personal and professional growth and development and so, at times, requires extended working hours. Our work often requires travel to client sites. The first-year base compensation for this role is ranges from $110,000 - $160,000 USD. At BCG, We Are Committed To Offering a Comprehensive Benefit Program That Includes Everything Our Employees And Their Families Need To Be Well And Live Life To The Fullest. We Pay The Full Cost Of Medical, Dental, And Vision Coverage For Employees â€“ And Their Eligible Family Members. * That's Zero Dollars In Premiums Taken From Employee Paychecks. All Our Plans Provide Best In Class Coverage Zero-dollar ($0) health insurance premiums for BCG employees, spouses, and children Low $10 (USD) copays for trips to the doctor, urgent care visits and prescriptions for generic drugs Dental coverage, including up to $5,000 in orthodontia benefits Vision insurance with coverage for both glasses and contact lenses annually Reimbursement for gym memberships and other fitness activities Fully vested Profit-Sharing Retirement Fund contributions made annually, whether you contribute or not, plus the option for employees to make personal contributions to a 401(k) plan Paid Parental Leave and other family benefits such as elective egg freezing, surrogacy, and adoption reimbursement Generous paid time off including 12 holidays per year, an annual office closure between Christmas and New Years, and 15 vacation days per year (earned at 1.25 days per month) Paid sick time on an as needed basis Employees, spouses, and children are covered at no cost. Employees share in the cost of domestic partner coverage. #BCGXjob Boston Consulting Group is an Equal Opportunity Employer. All qualified applicants will receive consideration for employment without regard to race, color, age, religion, sex, sexual orientation, gender identity / expression, national origin, disability, protected veteran status, or any other characteristic protected under national, provincial, or local law, where applicable, and those with criminal histories will be considered in a manner consistent with applicable state and local laws. BCG is an E - Verify Employer. Click here for more information on E-Verify.
        """)
    }
        
        
    ])
    return data
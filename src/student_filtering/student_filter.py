import pandas as pd
import re


class StudentFilter:

    REQUIRED_COLUMNS = [
        "student_id",
        "name",
        "email",
        "programme",
        "department",
        "level",
        "interests"
    ]

    def __init__(self, students_df: pd.DataFrame):

        self.students_df = students_df.copy()

        self._validate_columns()
        self._clean_data()

    # ==========================================================
    # VALIDATE STUDENT DATA
    # ==========================================================

    def _validate_columns(self):

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in self.students_df.columns
        ]

        if missing_columns:

            raise ValueError(
                f"Missing required Excel columns: {missing_columns}"
            )

    # ==========================================================
    # CLEAN STUDENT DATA
    # ==========================================================

    def _clean_data(self):

        for column in self.REQUIRED_COLUMNS:

            self.students_df[column] = (
                self.students_df[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        self.students_df["level"] = (
            self.students_df["level"]
            .str.lower()
            .str.strip()
        )

    # ==========================================================
    # NORMALISE TEXT
    # ==========================================================

    @staticmethod
    def _normalise_text(text):

        if text is None:
            return ""

        text = str(text).lower()

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # ==========================================================
    # KEYWORD MATCHING
    # ==========================================================

    @staticmethod
    def _contains_keyword(text, keyword):

        text = StudentFilter._normalise_text(text)
        keyword = StudentFilter._normalise_text(keyword)

        if not text or not keyword:
            return False

        return keyword in text

    # ==========================================================
    # GET TARGET ACADEMIC LEVELS
    # ==========================================================

    def _get_target_levels(self, target_audience):

        audience = self._normalise_text(
            target_audience
        )

        levels = []

        # Undergraduate
        if (
            "undergraduate" in audience
            or "undergraduate students" in audience
        ):
            levels.append("undergraduate")

        # Postgraduate
        if (
            "postgraduate" in audience
            or "post graduate" in audience
        ):
            levels.append("postgraduate")

        # Doctoral / PhD
        if (
            "doctoral" in audience
            or "phd" in audience
            or "research scholar" in audience
            or "research scholars" in audience
        ):
            levels.append("doctoral")

        return list(set(levels))

    # ==========================================================
    # EXPLICIT AUDIENCE ELIGIBILITY
    # ==========================================================

    def _check_explicit_eligibility(
        self,
        student,
        target_audience
    ):

        target_levels = self._get_target_levels(
            target_audience
        )

        student_level = self._normalise_text(
            student["level"]
        )

        # ------------------------------------------------------
        # If explicit audience specifies academic levels
        # ------------------------------------------------------

        if target_levels:

            if student_level in target_levels:

                return (
                    True,
                    (
                        "Student satisfies the explicit "
                        "academic-level requirement."
                    )
                )

            return (
                False,
                (
                    "Student does not satisfy the explicit "
                    "academic-level requirement."
                )
            )

        # ------------------------------------------------------
        # Explicit audience but no recognisable level
        # ------------------------------------------------------

        return (
            True,
            (
                "Poster has an explicit target audience, "
                "but no specific academic-level restriction "
                "was identified."
            )
        )

    # ==========================================================
    # RELEVANCE SCORE
    # ==========================================================

    def _calculate_relevance_score(
        self,
        student,
        metadata
    ):

        score = 0

        reasons = []

        poster_department = getattr(
            metadata,
            "department",
            ""
        )

        poster_keywords = getattr(
            metadata,
            "keywords",
            []
        )

        target_audience = getattr(
            metadata,
            "target_audience",
            ""
        )

        # ------------------------------------------------------
        # Student information
        # ------------------------------------------------------

        student_department = student[
            "department"
        ]

        student_programme = student[
            "programme"
        ]

        student_interests = student[
            "interests"
        ]

        # ======================================================
        # 1. DEPARTMENT MATCH
        # ======================================================

        if poster_department:

            if (
                self._contains_keyword(
                    student_department,
                    poster_department
                )
                or
                self._contains_keyword(
                    poster_department,
                    student_department
                )
            ):

                score += 5

                reasons.append(
                    "Department matches the poster department."
                )

        # ======================================================
        # 2. PROGRAMME VS POSTER KEYWORDS
        # ======================================================

        programme_matches = []

        for keyword in poster_keywords:

            if self._contains_keyword(
                student_programme,
                keyword
            ):

                programme_matches.append(
                    keyword
                )

                score += 4

        if programme_matches:

            programme_matches = list(
                dict.fromkeys(
                    programme_matches
                )
            )

            reasons.append(
                "Programme matches poster topics: "
                + ", ".join(programme_matches)
            )

        # ======================================================
        # 3. STUDENT INTERESTS VS POSTER KEYWORDS
        # ======================================================

        interest_matches = []

        student_interest_list = [
            interest.strip()
            for interest in student_interests.split(",")
            if interest.strip()
        ]

        for keyword in poster_keywords:

            for interest in student_interest_list:

                if (
                    self._contains_keyword(
                        interest,
                        keyword
                    )
                    or
                    self._contains_keyword(
                        keyword,
                        interest
                    )
                ):

                    interest_matches.append(
                        keyword
                    )

                    score += 3

                    break

        if interest_matches:

            interest_matches = list(
                dict.fromkeys(
                    interest_matches
                )
            )

            reasons.append(
                "Interests match poster topics: "
                + ", ".join(interest_matches)
            )

        # ======================================================
        # 4. RESEARCH / ACADEMIC RELEVANCE
        # ======================================================

        research_terms = [
            "research",
            "researcher",
            "researchers",
            "academic",
            "academics",
            "academician",
            "academicians",
            "scholar",
            "scholars",
            "phd",
            "doctoral"
        ]

        combined_student_text = (
            student_programme
            + " "
            + student_department
            + " "
            + student_interests
        )

        research_match = False

        for term in research_terms:

            if self._contains_keyword(
                combined_student_text,
                term
            ):

                research_match = True

                break

        if research_match:

            score += 3

            reasons.append(
                "Student profile indicates "
                "research or academic interest."
            )

        # ======================================================
        # 5. TARGET AUDIENCE ALIGNMENT
        # ======================================================

        audience_words = [
            word
            for word in self._normalise_text(
                target_audience
            ).split()
            if len(word) > 3
        ]

        audience_matches = []

        for word in audience_words:

            if (
                self._contains_keyword(
                    student_programme,
                    word
                )
                or
                self._contains_keyword(
                    student_department,
                    word
                )
                or
                self._contains_keyword(
                    student_interests,
                    word
                )
            ):

                audience_matches.append(
                    word
                )

                score += 2

        if audience_matches:

            audience_matches = list(
                dict.fromkeys(
                    audience_matches
                )
            )

            reasons.append(
                "Student profile aligns with "
                "target-audience terminology: "
                + ", ".join(audience_matches)
            )

        return score, reasons

    # ==========================================================
    # MAIN STUDENT FILTER
    # ==========================================================

    def filter_students(
        self,
        metadata,
        minimum_score=5
    ):

        output_columns = (
            self.REQUIRED_COLUMNS
            + [
                "relevance_score",
                "selection_reason"
            ]
        )

        results = []

        target_audience_type = (
            str(
                getattr(
                    metadata,
                    "target_audience_type",
                    ""
                )
            )
            .lower()
            .strip()
        )

        target_audience = getattr(
            metadata,
            "target_audience",
            ""
        )

        # ======================================================
        # CASE 1: EXPLICIT TARGET AUDIENCE
        # ======================================================

        if target_audience_type == "explicit":

            for _, student in self.students_df.iterrows():

                eligible, eligibility_reason = (
                    self._check_explicit_eligibility(
                        student,
                        target_audience
                    )
                )

                # Explicit eligibility is a HARD GATE.
                #
                # Example:
                # Poster = PhD Scholars
                # Undergraduate = rejected
                # Postgraduate = rejected
                # Doctoral = continue
                #
                # Interests cannot override this.

                if not eligible:
                    continue

                score, reasons = (
                    self._calculate_relevance_score(
                        student,
                        metadata
                    )
                )

                if score >= minimum_score:

                    all_reasons = [
                        eligibility_reason
                    ]

                    all_reasons.extend(
                        reasons
                    )

                    results.append(
                        {
                            **student.to_dict(),

                            "relevance_score": score,

                            "selection_reason":
                                " ".join(
                                    all_reasons
                                )
                        }
                    )

        # ======================================================
        # CASE 2: INFERRED TARGET AUDIENCE
        # ======================================================

        elif target_audience_type == "inferred":

            for _, student in self.students_df.iterrows():

                score, reasons = (
                    self._calculate_relevance_score(
                        student,
                        metadata
                    )
                )

                if score >= minimum_score:

                    all_reasons = [
                        (
                            "Target audience was inferred; "
                            "student selected based on "
                            "profile relevance."
                        )
                    ]

                    all_reasons.extend(
                        reasons
                    )

                    results.append(
                        {
                            **student.to_dict(),

                            "relevance_score": score,

                            "selection_reason":
                                " ".join(
                                    all_reasons
                                )
                        }
                    )

        # ======================================================
        # CASE 3: TARGET AUDIENCE NOT AVAILABLE
        # ======================================================

        else:

            return pd.DataFrame(
                columns=output_columns
            )

        # ======================================================
        # NO MATCHING STUDENTS
        # ======================================================

        if not results:

            return pd.DataFrame(
                columns=output_columns
            )

        # ======================================================
        # CREATE RESULT DATAFRAME
        # ======================================================

        result_df = pd.DataFrame(
            results
        )

        # Make sure every expected column exists.

        for column in output_columns:

            if column not in result_df.columns:

                result_df[column] = ""

        # ======================================================
        # SORT BY RELEVANCE
        # ======================================================

        result_df = result_df.sort_values(
            by="relevance_score",
            ascending=False
        )

        result_df = result_df.reset_index(
            drop=True
        )

        return result_df[
            output_columns
        ]
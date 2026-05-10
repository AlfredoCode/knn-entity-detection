class Mapper:
    """
    Bidirectional mapper between:
    - CNEC tags
    - Label Studio labels
    - Internal normalized entity schema
    """

    CNEC_TO_INTERNAL = {
        "p": "PersonalName",
        "P": "PersonalName",
        "ps": "PersonalName",
        "pf": "PersonalName",
        "pm": "PersonalName",
        "pd": "PersonalName",
        "pc": "PersonalName",
        "pp": "PersonalName",
        "pb": "PersonalName",
        "p_": "PersonalName",

        "g": "Location_General",
        "g_": "Location_General",
        "gu": "Location_ManMade",
        "gc": "Location_ManMade",
        "gr": "Location_ManMade",
        "gq": "Location_ManMade",

        "gs": "Location_Structure",
        "gp": "Location_Structure",

        "gh": "Location_Natural",
        "gl": "Location_Natural",
        "gt": "Location_Natural",


        "i": "Institution",
        "i_": "Institution",
        "ic": "Institution",
        "if": "Institution",
        "io": "Institution",
        "ia": "Institution",

        "o": "Object",
        "o_": "Object",
        "oa": "Object",
        "op": "Object",
        "om": "Object",
        "oe": "Object",
        "or": "Object",
        "oc": "Object",

        "t": "Time",
        "th": "Time",
        "ty": "Time",
        "tm": "Time",
        "td": "Time",
        "tf": "Time",

        "m": "Media",
        "mn": "Media",
        "ms": "Media",
        "mi": "Media",

        "a": "Address",
        "ah": "Address",
        "at": "Address",
        "az": "Address",
    }

    LABEL_STUDIO_TO_INTERNAL = {
        "per":    "PersonalName",
        "loc_c":  "Location_ManMade",
        "loc_n":  "Location_Natural",
        "loc_s":  "Location_Structure",
        "ins":    "Institution",
        "tim":    "Time",
        "med":    "Media",
        "obj_a":  "Object",
        "obj_p":  "Object",
        "groups": "O",
        "evt":    "O",
        "ide":    "O",
        "misc":   "O",
        "amb":    "O",
    }



    def cnec_to_bioes(self, tokens, entities):
        labels = ["O"] * len(tokens)

        for start, end, cnec_label in entities:
            internal = self.CNEC_TO_INTERNAL.get(cnec_label, "O")
            if internal == "O":
                continue

            length = end - start + 1

            if length == 1:
                labels[start] = f"S-{internal}"
            else:
                labels[start] = f"B-{internal}"
                for i in range(start + 1, end):
                    labels[i] = f"I-{internal}"
                labels[end] = f"E-{internal}"

        return list(zip(tokens, labels))

    

    def explain_cnec(self, tag: str):
        return self.CNEC_TO_INTERNAL.get(tag, "UNKNOWN")

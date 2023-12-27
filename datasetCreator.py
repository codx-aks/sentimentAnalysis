import pandas as pd
import random

# Function to generate synthetic call transcripts
def generate_call_transcripts(num_entries=10000):
    texts = []
    labels = []

    quality_issues = [
        "damaged product", "poor quality", "defective item",
        "missing parts", "wrong color", "expired product",
        "incomplete set", "mismatched product", "bad odor",
        "stale product", "counterfeit item", "product contamination",
        "packaging issue", "wrong size", "subpar materials",
        "low durability", "ineffective product", "not as advertised",
        "allergic reaction", "unpleasant taste", "inadequate instructions",
        "difficult assembly", "missing accessories", "poor craftsmanship",
        "unsatisfactory design", "faulty mechanism", "unreliable performance",
        "unintuitive controls", "unstable construction", "inconsistent manufacturing",
        "misleading product description", "uncomfortable fit", "unreasonable weight",
        "inconvenient return process", "refund not processed timely",
        "refund rejection", "difficulty canceling order", "customer not informed of delays",
        "product not meeting expectations", "difficulty contacting customer support",
        "unresolved product issue", "incomplete product information", "unresponsive seller"
    ]

    delivery_issues = [
        "late delivery", "missing items", "delivery problem",
        "package not received", "wrong item delivered",
        "delivery to wrong address", "package damaged in transit",
        "delivery delay", "order lost in transit", "package stolen",
        "incorrect tracking information", "delivery to neighbor",
        "delivery to old address", "delivery to closed business",
        "delivery to vacation address", "delivery outside specified hours",
        "delivery to inaccessible location", "delivery to unsecure location",
        "delivery to unattended address", "customs clearance issue",
        "delivery to unsafe area", "delivery to unverified address",
        "delivery to construction site", "delivery to restricted zone",
        "delivery to hazardous location", "delivery to secure facility",
        "delivery to military base", "delivery to government office",
        "delivery to remote area", "delivery to diplomatic premises",
        "delivery to correctional facility", "delivery to healthcare institution",
        "delivery to educational institution", "delivery to religious institution",
        "delivery to research facility", "delivery to entertainment venue",
        "incorrect delivery charge", "unexpected delivery charges", "incorrect delivery time",
        "misdelivered package", "wrong quantity delivered", "delivery confirmation issues",
        "multiple delivery attempts", "incomplete delivery information", "inconvenient pickup process"
    ]

    service_issues = [
        "unhelpful service", "rude behavior", "unsatisfactory support",
        "long wait times", "incorrect information", "unresolved complaint",
        "ignored customer", "no response to queries", "unfriendly staff",
        "miscommunication", "untrained staff", "unresponsive helpline",
        "poor customer handling", "difficulty reaching support",
        "lack of empathy", "untrained support agents", "unreliable customer assistance",
        "unprofessional conduct", "difficulty escalating issues", "confusing customer service policies",
        "unapologetic service", "impersonal assistance", "lack of follow-up",
        "insufficient guidance", "unsympathetic response", "uncooperative attitude",
        "lack of accountability", "unwillingness to help", "lack of urgency",
        "failure to address concerns", "uninformed staff", "inflexible policies",
        "inconsistent responses", "unavailable supervisors", "lack of empowerment",
        "misguided troubleshooting", "uncoordinated support", "indifferent service",
        "language barrier in support", "support not available in preferred language",
        "support not accessible", "unhelpful chat support", "unresponsive email support",
        "escalation process not clear", "support not 24/7", "difficulty accessing self-help resources",
        "you are not doing your work properly", "answer me correctly",
        "incompetent support", "lack of resolution", "customer service failure"
    ]

    billing_issues = [
        "billing error", "overcharged", "payment problem",
        "double billing", "subscription issue", "unauthorized charges",
        "billing discrepancy", "wrong invoice", "auto-renewal issue",
        "refund not processed", "billing cycle confusion", "hidden fees",
        "billing fraud", "subscription not canceled", "upgrade/downgrade issues",
        "charged for canceled service", "billing after account closure",
        "no receipt provided", "incorrect payment method charged",
        "refund process delay", "billing statement error", "refund rejection",
        "billing system glitch", "unexplained charges", "service not rendered",
        "service not reflected in bill", "billing dispute", "late payment charges",
        "penalty for early payment", "unauthorized payment method change",
        "misapplied payment", "overdue notice error", "credit card processing issue",
        "payment gateway failure", "transaction declined", "fraudulent transaction",
        "payment confirmation delay", "bank transfer issues", "payment link not working",
        "currency conversion problem", "incomplete payment information", "invalid payment method",
        "refund not credited", "non-reversal of charges", "billing support not accessible"
    ]

    technical_issues = [
        "website issue", "app crash", "technical problem",
        "login failure", "lost data", "functionality failure",
        "unresponsive interface", "browser compatibility issue", "slow performance",
        "unexpected errors", "broken links", "malware detection",
        "security vulnerability", "incorrect calculations", "payment gateway failure",
        "server downtime", "system maintenance issues", "data breach",
        "database corruption", "data synchronization problem", "integration failure",
        "API connectivity issue", "third-party service failure", "compatibility issues",
        "software bug", "incorrect results", "syncing issues",
        "device compatibility problem", "login authentication failure", "server overload",
        "disconnection problems", "data not saving", "incorrect data display",
        "file upload/download failure", "image rendering issues", "video playback problems",
        "audio issues", "unreliable notifications", "software freeze", "memory leak",
        "unavailable features", "unexplained system behavior", "incorrect system response",
        "lack of user guidance", "confusing interface", "misleading prompts"
    ]

    product_availability = [
        "item not restocked", "product recall", "inconsistent inventory",
        "misleading product availability", "unpredictable stock updates",
        "fluctuating stock levels", "inaccurate stock information", "product not on shelves",
        "long backorder wait times", "demand exceeds supply", "supply chain disruptions",
        "inability to meet demand", "difficulty obtaining products", "supplier issues",
        "manufacturing delays", "raw material shortage", "production halt",
        "logistic challenges", "import/export restrictions", "customs clearance delay",
        "trade disputes", "supplier bankruptcy", "unforeseen market demand",
        "quality control problems", "product expiration", "seasonal availability issues",
        "shipping restrictions", "transportation issues", "global events affecting supply chain",
        "supplier mismanagement", "cancellation of production", "unanticipated demand spike",
        "overwhelming demand", "logistics coordination problems", "supply chain bottlenecks",
        "market fluctuations affecting availability", "distribution network issues",
        "warehouse capacity challenges", "unexpected demand patterns", "limited production capacity",
        "stock allocation problems", "unstable demand forecasting", "supply chain visibility issues",
        "supplier relationship problems", "difficulty sourcing raw materials", "unreliable suppliers",
        "shortage of skilled labor", "global economic factors impacting supply chain",
        "dependency on single suppliers", "geopolitical issues affecting supply chain",
        "lack of redundancy in supply chain", "delayed product launches due to supply chain",
        "supply chain risks", "unstable international trade environment", "currency exchange challenges",
        "high shipping costs", "customs clearance complexities", "border crossing delays",
        "tariff-related challenges", "logistics partners failure", "unplanned production halts"
    ]

    for _ in range(num_entries):
        rando=random.random()
        # if rando" < 0.17:
        #     text = f"I have a {random.choice(quality_issues)} issue."
        #     label = "Quality Issues"
        # elif rando < 0.34:
        #     text = f"My {random.choice(delivery_issues)} is causing problems."
        #     label = "Delivery Issues"
        # elif rando < 0.51:
        #     text = f"Your {random.choice(service_issues)} is unacceptable."
        #     label = "Service Issues"
        # elif rando < 0.67:
        #     text = f"There's a {random.choice(billing_issues)} in my bill."
        #     label = "Billing Issues"
        # elif rando < 0.83:
        #     text = f"I'm facing {random.choice(technical_issues)} on your platform."
        #     label = "Technical Issues"
        # else:
        #     text = f"The {random.choice(product_availability)} issue is frustrating."
        #     label = "Product Availability
        if rando < 0.17:
            ran=random.random()
            if ran < 0.20:
                text = f"I have a {random.choice(quality_issues)} issue."
            elif ran < 0.36:
                text = f"My {random.choice(quality_issues)} is causing problems."
            elif ran < 0.52:
                text = f"Your {random.choice(quality_issues)} is unacceptable."
            elif ran < 0.68:
                text = f"There's a {random.choice(quality_issues)} in my product."
            elif ran < 0.84:
                text = f"I'm facing {random.choice(quality_issues)} on your product."
            else:
                text = f"The {random.choice(quality_issues)} issue is frustrating."
            label = "Quality Issues"

        elif rando < 0.34:
            ran=random.random()
            if ran < 0.16:
                text = f"I have a {random.choice(delivery_issues)} issue."
            elif ran < 0.36:
                text = f"My {random.choice(delivery_issues)} is causing problems."
            elif ran < 0.52:
                text = f"Your {random.choice(delivery_issues)} is unacceptable."
            elif ran < 0.68:
                text = f"There's a {random.choice(delivery_issues)} for my product."
            elif ran < 0.84:
                text = f"I'm facing {random.choice(delivery_issues)} on your product."
            else:
                text = f"The {random.choice(delivery_issues)} issue is frustrating."

            label = "Delivery Issues"
        elif rando < 0.51:
            ran = random.random()
            if ran < 0.16:
                text = f"I have a {random.choice(service_issues)} issue."
            elif ran < 0.32:
                text = f"My {random.choice(service_issues)} is causing problems."
            elif ran < 0.52:
                text = f"Your {random.choice(service_issues)} is unacceptable."
            elif ran < 0.68:
                text = f"There's a {random.choice(service_issues)} in your company."
            elif ran < 0.84:
                text = f"I'm facing a {random.choice(service_issues)} ."
            else:
                text = f"The {random.choice(service_issues)} issue is frustrating."
            label = "Service Issues"
        elif rando < 0.67:
            ran = random.random()
            if ran < 0.16:
                text = f"I have a {random.choice(billing_issues)} issue."
            elif ran < 0.32:
                text = f"My {random.choice(billing_issues)} is causing problems."
            elif ran < 0.48:
                text = f"Your {random.choice(billing_issues)} is unacceptable."
            elif ran < 0.68:
                text = f"There's a {random.choice(billing_issues)} in my bill."
            elif ran < 0.84:
                text = f"I'm facing {random.choice(billing_issues)} on your product."
            else:
                text = f"The {random.choice(billing_issues)} issue is frustrating."
            label = "Billing Issues"

        elif rando < 0.83:
            ran = random.random()
            if ran < 0.16:
                text = f"I have a {random.choice(technical_issues)} issue."
            elif ran < 0.32:
                text = f"My {random.choice(technical_issues)} is causing problems."
            elif ran < 0.48:
                text = f"Your {random.choice(technical_issues)} is unacceptable."
            elif ran < 0.64:
                text = f"There's a {random.choice(technical_issues)} in my product."
            elif ran < 0.84:
                text = f"I'm facing {random.choice(technical_issues)} on your platform."
            else:
                text = f"The {random.choice(technical_issues)} issue is frustrating."
            label = "Technical Issues"
        else:
            ran=random.random()
            if ran < 0.16:
                text = f"I have a {random.choice(product_availability)} issue."
            elif ran < 0.32:
                text = f"My {random.choice(product_availability)} is causing problems."
            elif ran < 0.48:
                text = f"Your {random.choice(product_availability)} is unacceptable."
            elif ran < 0.64:
                text = f"There's a {random.choice(product_availability)} in my product."
            elif ran < 0.80:
                text = f"I'm facing {random.choice(product_availability)} for your product."
            else:
                text = f"The {random.choice(product_availability)} issue is frustrating."

            label = "Product Availability"

        texts.append(text)
        labels.append(label)

    return pd.DataFrame({"Text": texts, "Label": labels})


dataset = generate_call_transcripts(10000)

dataset.to_csv("issue_dataset.csv", index=False)

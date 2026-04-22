import pandas as pd
import random

def generate_reviews(n=3000):
    positive_phrases = [
        "absolutely love", "great experience", "highly recommend", "wonderful", "fantastic",
        "best I have ever used", "exceeded my expectations", "superb quality", "works perfectly",
        "very satisfied", "amazing product", "delighted with", "excellent customer service",
        "five stars", "top notch"
    ]
    
    negative_phrases = [
        "terrible", "waste of money", "do not buy", "awful", "horrible experience",
        "broke immediately", "very disappointed", "worst product", "poor quality",
        "stopped working", "useless", "complete garbage", "frustrating", "bad customer service",
        "never buying again", "refund"
    ]
    
    neutral_phrases = [
        "it is okay", "average", "nothing special", "does the job", "as expected",
        "middle of the road", "fine", "not bad not good", "standard", "acceptable",
        "three stars", "could be better", "meets requirements", "fair price", "decent"
    ]
    
    subjects = ["This product", "The service", "The item", "It", "The quality", "Everything", "The delivery"]
    
    fillers = [
        " overall.", " to be honest.", " in my opinion.", ".", " honestly.", 
        " for the price.", " so far.", " right now.", " exactly."
    ]

    data = []
    
    for _ in range(n // 3):
        text = f"{random.choice(subjects)} is {random.choice(positive_phrases)}{random.choice(fillers)}"
        data.append({"text": text, "sentiment": "positive"})
        
        text = f"{random.choice(subjects)} was a {random.choice(negative_phrases)}{random.choice(fillers)}"
        if "is" not in text and "was" not in text:
            text = f"I think {random.choice(subjects).lower()} is {random.choice(negative_phrases)}."
        data.append({"text": text, "sentiment": "negative"})
        
        text = f"{random.choice(subjects)} is {random.choice(neutral_phrases)}{random.choice(fillers)}"
        data.append({"text": text, "sentiment": "neutral"})
        
    df = pd.DataFrame(data)
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv("data/reviews.csv", index=False)
    print(f"Generated {len(df)} synthetic reviews in data/reviews.csv")

if __name__ == "__main__":
    generate_reviews()

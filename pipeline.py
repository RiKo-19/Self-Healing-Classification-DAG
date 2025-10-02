import logging
from typing import TypedDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from langgraph.graph import StateGraph, END


# Load Fine-tuned Model

model_path = "./fine_tuned_imdb"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Fine-tuned classifier
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)

# Label mapping (since your model outputs 0/1)
label_map = {0: "NEGATIVE", 1: "POSITIVE"}


# Load Backup Model (Zero-Shot Classifier)

backup_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ["POSITIVE", "NEGATIVE"]


# Setup Logging

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# Define State Schema

class State(TypedDict, total=False):
    input: str
    prediction: str
    confidence: float
    fallback_needed: bool
    backup_prediction: str
    backup_confidence: float
    final_label: str


# Define DAG Nodes

def inference_node(state: State) -> State:
    """Runs classification on input text."""
    text = state["input"]
    results = classifier(text, truncation=True, max_length=256)[0]
    
    # Pick best label
    best = max(results, key=lambda x: x["score"])
    label_id = int(best["label"].split("_")[-1])
    label = label_map[label_id]
    confidence = best["score"]
    
    logging.info(f"[InferenceNode] Input: {text} | Predicted: {label} | Confidence: {confidence:.2f}")
    return {"prediction": label, "confidence": confidence, "input": text}


def confidence_check_node(state: State) -> State:
    """Checks confidence threshold."""
    conf = state["confidence"]
    if conf < 0.70:
        logging.info(f"[ConfidenceCheckNode] Low confidence ({conf:.2f}). Triggering fallback...")
        return {"fallback_needed": True, **state}
    else:
        return {"fallback_needed": False, **state}


def fallback_node(state: State) -> State:
    """Backup model prediction + user clarification if needed."""
    if state["fallback_needed"]:
        text = state["input"]
        
        # Backup model prediction
        backup = backup_classifier(text, candidate_labels=candidate_labels)
        backup_label = backup["labels"][0]
        backup_conf = backup["scores"][0]
        
        logging.info(f"[FallbackNode] Backup Model Prediction: {backup_label} | Confidence: {backup_conf:.2f}")
        print(f"[FallbackNode] Backup Model suggests: {backup_label} (conf: {backup_conf:.2f})")
        
        # If backup agrees with original → accept backup
        if backup_label == state["prediction"]:
            logging.info(f"[FallbackNode] Backup agrees with main model → Accepting {backup_label}")
            return {"final_label": backup_label, "backup_prediction": backup_label, "backup_confidence": backup_conf, **state}
        
        # If they disagree → ask user
        print(f"[FallbackNode] Main model = {state['prediction']} (conf: {state['confidence']:.2f}), "
              f"Backup model = {backup_label} (conf: {backup_conf:.2f})")
        user_input = input("They disagree. Please clarify (POSITIVE/NEGATIVE): ").strip().upper()
        
        if user_input in ["POSITIVE", "NEGATIVE"]:
            final_label = user_input
        else:
            print("Invalid input. Keeping backup model prediction.")
            final_label = backup_label
        
        logging.info(f"[FallbackNode] User clarified → Final label: {final_label}")
        return {"final_label": final_label, "backup_prediction": backup_label, "backup_confidence": backup_conf, **state}
    
    else:
        logging.info(f"[FallbackNode] Accepted prediction: {state['prediction']}")
        return {"final_label": state["prediction"], **state}


# Build LangGraph Workflow

graph = StateGraph(State)
graph.add_node("InferenceNode", inference_node)
graph.add_node("ConfidenceCheckNode", confidence_check_node)
graph.add_node("FallbackNode", fallback_node)

# Define flow with entrypoint
graph.set_entry_point("InferenceNode")
graph.add_edge("InferenceNode", "ConfidenceCheckNode")
graph.add_edge("ConfidenceCheckNode", "FallbackNode")
graph.add_edge("FallbackNode", END)

# Compile graph
workflow = graph.compile()


# CLI Loop

print("Sentiment Classification CLI with Backup Fallback (type 'quit' to exit)")
while True:
    text = input("\nEnter a review: ")
    if text.lower() == "quit":
        print("Exiting...")
        break
    
    state: State = {"input": text}
    result = workflow.invoke(state)
    
    print(f"Final Label: {result['final_label']} "
          f"(main conf: {result['confidence']:.2f}, "
          f"backup: {result.get('backup_prediction', 'N/A')} "
          f"({result.get('backup_confidence', 0):.2f}))")

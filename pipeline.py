from pathlib import Path

from data_ingestion import DataIngestion
from train import CreditScoreTrainer
from evaluation import Evaluator


class CreditScorePipeline:

    def __init__(self, accuracy_threshold=0.70):
        self.accuracy_threshold = accuracy_threshold

        self.base_dir = Path(__file__).parent

        self.ingestor = DataIngestion(
            input_path=self.base_dir / "data_D.csv",
            output_dir=self.base_dir / "ingested"
        )

        self.trainer = CreditScoreTrainer()
        self.evaluator = Evaluator()

    def execute(self):

        print("===== Credit Score ML Pipeline =====")

        # Step 1 - Data Ingestion
        print("\n[STEP 1] Data Ingestion")
        data_path = self.ingestor.run()

        # Step 2 - Training
        print("\n[STEP 2] Training")
        best_name, best_pipeline, x_test, y_test = self.trainer.run(str(data_path))

        # Step 3 - Evaluation
        print("\n[STEP 3] Evaluation")
        acc, precision, recall, f1, report = self.evaluator.evaluate(
            x_test=x_test,
            y_test=y_test,
            model=best_pipeline,
        )

        # Deployment Decision
        print("\n--- Deployment Approval Decision ---")

        if acc >= self.accuracy_threshold:
            print(f"✅ Model APPROVED (accuracy={acc:.4f})")
        else:
            print(f"❌ Model REJECTED (accuracy={acc:.4f} < {self.accuracy_threshold})")

        return acc, best_name


if __name__ == "__main__":
    pipeline = CreditScorePipeline()
    pipeline.execute()
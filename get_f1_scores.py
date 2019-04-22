import pandas as pd 
import torch
from ast import literal_eval
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import argparse
import sys 


class GAPScorer(object):
    """
    Container class for storing scores, and generating evaluation metrics.
    From Google. 
    Attributes:
    true_positives: Tally of true positives seen.
    false_positives: Tally of false positives seen.
    true_negatives: Tally of true negatives seen.
    false_negatives: Tally of false negatives seen.
    """
    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0

    def recall(self):
        """Calculates recall based on the observed scores.
        Returns:
          float, the recall.
        """
        numerator = self.true_positives
        denominator = self.true_positives + self.false_negatives
        return 100.0 * numerator / denominator if denominator else 0.0

    def precision(self):
        """Calculates precision based on the observed scores.
        Returns:
          float, the precision.
        """
        numerator = self.true_positives
        denominator = self.true_positives + self.false_positives
        return 100.0 * numerator / denominator if denominator else 0.0

    def f1(self):
        """Calculates F1 based on the observed scores.
        Returns:
          float, the F1 score.
        """
        recall = self.recall()
        precision = self.precision()

        numerator = 2 * precision * recall
        denominator = precision + recall
        return numerator / denominator if denominator else 0.0

    def get_metric(self, reset=False):
        recall = self.recall()
        precision = self.precision()
        f1 = self.f1()
        if reset:
            self.true_positives = 0
            self.false_positives = 0
            self.true_negatives = 0
            self.false_negatives = 0
        return f1

    def __call__(self, predictions, labels):
        """Score the system annotations against gold.
        Args:
        predictiosn: torch tensor of batch x 3,  doesn't have to be one-hot vector yet. 
        system_annotations: batch x 3 Torch numpy list
        Returns:
        None
        """
        pred_indices = torch.max(predictions, dim=1)[1].view(-1, 1)
        num_classes = 3
        one_hot_logits = (pred_indices == torch.arange(num_classes).reshape(1, num_classes)).float()
        predictions = one_hot_logits[:,:2].numpy()
        labels = labels[:,:2].numpy()
        b_size = predictions.shape[0]
        for i in range(b_size):
            for j in range(2):
                pred = predictions[i][j].item()
                gold = labels[i][j].item()
                if gold and pred:
                  self.true_positives += 1
                elif not gold and pred:
                  self.false_positives += 1
                elif not gold and not pred:
                  self.true_negatives += 1
                elif gold and not pred:
                  self.false_negatives += 1
        return


# start merging in the GAP scores. 
def get_accuracy_and_f1_scores_gap(path_to_run, task_name):
	try:
		val = pd.read_csv(path_to_run+task_name+"_val.tsv", sep="\t")
		labels = val["true_label"].apply(lambda x: literal_eval(x))
		labels = torch.LongTensor(labels.tolist())
		labels = labels.squeeze(dim=1)
		labels = torch.argmax(labels, dim=1)
		preds = val["prediction"].tolist()
		print("Validation results")
		print("Accuracy:")
		print(accuracy_score(preds, labels))
		print('F1 score:')
		scorer = GAPScorer()
		def one_hot_v(batch, depth=3):
			ones = torch.sparse.torch.eye(depth)
			return ones.index_select(0,batch)
		predictions = torch.LongTensor(preds)
		one_hot_preds = one_hot_v(predictions)
		one_hot_labels = one_hot_v(torch.LongTensor(labels))
		# make the predictions the actual one hot label
		scorer(one_hot_preds,one_hot_labels)
		print(scorer.get_metric(reset=False))
		print("confusion matrix")
		print("[TN, FP]")
		print("[FN, TP]")
		print(confusion_matrix(preds, labels))
	except Exception as e:
		pass
	val = pd.read_csv(path_to_run+task_name+"_test.tsv", sep="\t")
	labels = val["true_label"].apply(lambda x: literal_eval(x))
	labels = torch.LongTensor(labels.tolist())
	labels = labels.squeeze(dim=1)
	labels = torch.argmax(labels, dim=1)
	preds = val["prediction"].tolist()
	print("Test results")
	print("Accuracy:")
	print(accuracy_score(preds, labels))
	print('F1 score:')
	scorer = GAPScorer()
	def one_hot_v(batch, depth=3):
		ones = torch.sparse.torch.eye(depth)
		return ones.index_select(0,batch)
	predictions = torch.LongTensor(preds)
	one_hot_preds = one_hot_v(predictions)
	one_hot_labels = one_hot_v(torch.LongTensor(labels))
	# make the predictions the actual one hot label
	scorer(one_hot_preds,one_hot_labels)
	print(scorer.get_metric(reset=False))
	print("confusion matrix")
	print("[TN, FP]")
	print("[FN, TP]")
	print(confusion_matrix(preds, labels))
	
def get_accuracy_and_f1_scores(path_to_run, task_name):
	try:
		val = pd.read_csv(path_to_run+task_name+"_val.tsv", sep="\t")
		labels = val["true_label"].apply(lambda x: literal_eval(x))
		labels = torch.LongTensor(labels.tolist())
		labels = labels.squeeze(dim=1)
		labels = torch.argmax(labels, dim=1)
		preds = val["prediction"].tolist()
		if task_name == 'ultrafine-balanced':
			# now we only get finer statistics
			tagmask = val["category"].apply(lambda x: literal_eval(x)).tolist()
			tagmask = torch.LongTensor(tagmask)
			tagmask = torch.argmax(tagmask, dim=1)
			indices = (tagmask == 2).nonzero().squeeze(-1).tolist()
			preds = val["prediction"][indices]
			labels = labels[indices]
		print("Validation results")
		print("Accuracy:")
		print(accuracy_score(preds, labels))
		print('F1 score:')
		print(f1_score(preds, labels))
		print("confusion matrix")
		print("[TN, FP]")
		print("[FN, TP]")
		print(confusion_matrix(preds, labels))
	except Exception as e:
		pass
	val = pd.read_csv(path_to_run+task_name+"_test.tsv", sep="\t")
	labels = val["true_label"].apply(lambda x: literal_eval(x))
	labels = torch.LongTensor(labels.tolist())
	labels = labels.squeeze(dim=1)
	labels = torch.argmax(labels, dim=1)
	preds = val["prediction"].tolist()
	if task_name == 'ultrafine-balanced':
		# now we only get finer statistics
		tagmask = val["category"].apply(lambda x: literal_eval(x)).tolist()
		tagmask = torch.LongTensor(tagmask)
		tagmask = torch.argmax(tagmask, dim=1)
		indices = (tagmask == 2).nonzero().squeeze(-1).tolist()
		preds = val["prediction"][indices]
		labels = labels[indices]
	print("Test results")
	print("Accuracy:")
	print(accuracy_score(preds, labels))
	print('F1 score:')
	print(f1_score(preds, labels))
	print("confusion matrix")
	print("[TN, FP]")
	print("[FN, TP]")
	print(confusion_matrix(preds, labels))
	

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default=4,
                        help="Number of parallel processes to use.")
    parser.add_argument('--inputs', type=str, nargs="+",
                        help="Input JSON files.")
    args = parser.parse_args(args)
    if args.task_name == 'gap-coreference':
    	get_accuracy_and_f1_scores_gap(args.inputs[0], args.task_name)
    else:
    	get_accuracy_and_f1_scores(args.inputs[0], args.task_name)

if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit(0)
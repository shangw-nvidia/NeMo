import os
import json
import re
import collections

from nemo.utils.exp_logging import get_logger

# Dimension of the embedding for intents, slots and categorical slot values in
# the schema. Should be equal to BERT's hidden_size.
EMBEDDING_DIMENSION = 768
# Maximum allowed number of categorical trackable slots for a service.
MAX_NUM_CAT_SLOT = 6
# Maximum allowed number of non-categorical trackable slots for a service.
MAX_NUM_NONCAT_SLOT = 12
# Maximum allowed number of values per categorical trackable slot.
MAX_NUM_VALUE_PER_CAT_SLOT = 11
# Maximum allowed number of intents for a service.
MAX_NUM_INTENT = 4
STR_DONTCARE = "dontcare"
# The maximum total input sequence length after WordPiece tokenization.
DEFAULT_MAX_SEQ_LENGTH = 128

# These are used to represent the status of slots (off, active, dontcare) and
# intents (off, active) in dialogue state tracking.
STATUS_OFF = 0
STATUS_ACTIVE = 1
STATUS_DONTCARE = 2

FILE_RANGES = {
	"dstc8_single_domain": {
		"train": range(1, 44),
		"dev": range(1, 8),
		"test": range(1, 12)
	},
	"dstc8_multi_domain": {
		"train": range(44, 128),
		"dev": range(8, 21),
		"test": range(12, 35)
	},
	"dstc8_all": {
		"train": range(1, 3),
		"dev": range(1, 3),
		"test": range(1, 3)
	}
	# "dstc8_all": {
	# 	"train": range(1, 128),
	# 	"dev": range(1, 21),
	# 	"test": range(1, 35)
	# }
}

DATABASE_EXISTS_TMP = '{} dataset has already been processed and stored at {}'
MODE_EXISTS_TMP = \
	'{} mode of {} dataset has already been processed and stored at {}'

logger = get_logger('')


def if_exist(outfold, files):
	if not os.path.exists(outfold):
		return False
	for file in files:
		if not os.path.exists(f'{outfold}/{file}'):
			return False
	return True


class StateTrackingSGDDataDesc:
	""" Convert the raw data to the standard format supported by
	StateTrackingSGDData.
	TODO: Update here

	By default, the None label for slots is 'O'.

	JointIntentSlotDataset requires two files:

		input_file: file to sequence + label.
			the first line is header (sentence [tab] label)
			each line should be [sentence][tab][label]

		slot_file: file to slot labels, each line corresponding to
			slot labels for a sentence in input_file. No header.

	To keep the mapping from label index to label consistent during
	training and inferencing, we require the following files:
		dicts.intents.csv: each line is an intent. The first line
			corresponding to the 0 intent label, the second line
			corresponding to the 1 intent label, and so on.

		dicts.slots.csv: each line is a slot. The first line
			corresponding to the 0 slot label, the second line
			corresponding to the 1 slot label, and so on.

	Args:
		data_dir (str): the directory of the dataset
		do_lower_case (bool): whether to set your dataset to lowercase
		dataset_name (str): the name of the dataset. If it's a dataset
			that follows the standard JointIntentSlotDataset format,
			you can set the name as 'default'.
		none_slot_label (str): the label for slots that aren't indentified
			defaulted to 'O'
		pad_label (int): the int used for padding. If set to -1,
			 it'll be set to the whatever the None label is.

	"""

	def __init__(self,
				 data_dir,
				 tokenizer,
				 task_name,
				 dataset_split,
				 do_lower_case=False,
				 dataset_name='default',
				 none_slot_label='O',
				 pad_label=-1,
				 max_seq_length=50,
				 modes=['train', 'eval'],
				 log_data_warnings=False):
		if dataset_name == 'sgd':
			self.data_dir = process_sgd(data_dir,
										do_lower_case,
										dataset_name=dataset_name,
										max_seq_length=max_seq_length,
										task_name=task_name,
										tokenizer=tokenizer,
										dataset_split=dataset_split,
										modes=modes,
										log_data_warnings=log_data_warnings)
		else:
			if not if_exist(data_dir, ['dialogues.tsv']):
				raise FileNotFoundError(
					"Make sure that your data follows the standard format "
					"supported by StateTrackerDataset. Your data must "
					"contain dialogues.tsv.")
			self.data_dir = data_dir

	# Changed here
	# self.intent_dict_file = self.data_dir + '/dict.intents.csv'
	# self.slot_dict_file = self.data_dir + '/dict.slots.csv'
	# self.num_intents = len(get_vocab(self.intent_dict_file))
	# slots = label2idx(self.slot_dict_file)
	# self.num_slots = len(slots)
	#
	# for mode in ['train', 'test', 'eval']:
	#
	# 	if not if_exist(self.data_dir, [f'{mode}.tsv']):
	# 		logger.info(f' Stats calculation for {mode} mode'
	# 					f' is skipped as {mode}.tsv was not found.')
	# 		continue
	#
	# 	slot_file = f'{self.data_dir}/{mode}_slots.tsv'
	# 	with open(slot_file, 'r') as f:
	# 		slot_lines = f.readlines()
	#
	# 	input_file = f'{self.data_dir}/{mode}.tsv'
	# 	with open(input_file, 'r') as f:
	# 		input_lines = f.readlines()[1:]  # Skipping headers at index 0
	#
	# 	if len(slot_lines) != len(input_lines):
	# 		raise ValueError(
	# 			"Make sure that the number of slot lines match the "
	# 			"number of intent lines. There should be a 1-1 "
	# 			"correspondence between every slot and intent lines.")
	#
	# 	dataset = list(zip(slot_lines, input_lines))
	#
	# 	raw_slots, queries, raw_intents = [], [], []
	# 	for slot_line, input_line in dataset:
	# 		slot_list = [int(slot) for slot in slot_line.strip().split()]
	# 		raw_slots.append(slot_list)
	# 		parts = input_line.strip().split()
	# 		raw_intents.append(int(parts[-1]))
	# 		queries.append(' '.join(parts[:-1]))
	#
	# 	infold = input_file[:input_file.rfind('/')]
	#
	# 	logger.info(f'Three most popular intents during {mode}ing')
	# 	total_intents, intent_label_freq = get_label_stats(
	# 		raw_intents, infold + f'/{mode}_intent_stats.tsv')
	# 	merged_slots = itertools.chain.from_iterable(raw_slots)
	#
	# 	logger.info(f'Three most popular slots during {mode}ing')
	# 	slots_total, slots_label_freq = get_label_stats(
	# 		merged_slots, infold + f'/{mode}_slot_stats.tsv')
	#
	# 	if mode == 'train':
	# 		self.slot_weights = calc_class_weights(slots_label_freq)
	# 		logger.info(f'Slot weights are - {self.slot_weights}')
	#
	# 		self.intent_weights = calc_class_weights(intent_label_freq)
	# 		logger.info(f'Intent weights are - {self.intent_weights}')
	#
	# 	logger.info(f'Total intents - {total_intents}')
	# 	logger.info(f'Intent label frequency - {intent_label_freq}')
	# 	logger.info(f'Total Slots - {slots_total}')
	# 	logger.info(f'Slots label frequency - {slots_label_freq}')
	#
	# if pad_label != -1:
	# 	self.pad_label = pad_label
	# else:
	# 	if none_slot_label not in slots:
	# 		raise ValueError(f'none_slot_label {none_slot_label} not '
	# 						 f'found in {self.slot_dict_file}.')
	# 	self.pad_label = slots[none_slot_label]


class Dstc8DataProcessor(object):
	"""Data generator for dstc8 dialogues."""

	def __init__(self,
				 dstc8_data_dir,
				 train_file_range,
				 dev_file_range,
				 test_file_range,
				 tokenizer,
				 max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
				 log_data_warnings=False):
		self.dstc8_data_dir = dstc8_data_dir
		self._log_data_warnings = log_data_warnings
		self._file_ranges = {
			"train": train_file_range,
			"dev": dev_file_range,
			"test": test_file_range,
		}
		# BERT tokenizer
		self._tokenizer = tokenizer
		self._max_seq_length = max_seq_length

	def get_dialog_examples(self, dataset):
		"""Return a list of `InputExample`s of the data splits' dialogues.

		Args:
		  dataset: str. can be "train", "dev", or "test".

		Returns:
		  examples: a list of `InputExample`s.
		"""
		dialog_paths = [
			os.path.join(self.dstc8_data_dir, dataset,
						 "dialogues_{:03d}.json".format(i))
			for i in self._file_ranges[dataset]
		]
		dialogs = load_dialogues(dialog_paths)
		schema_path = os.path.join(self.dstc8_data_dir, dataset, "schema.json")
		schemas = Schema(schema_path)

		examples = []
		for dialog_idx, dialog in enumerate(dialogs):
			if dialog_idx % 1000 == 0:
				logger.info(f'Processed {dialog_idx} dialogs.')
			examples.extend(
				self._create_examples_from_dialog(dialog, schemas, dataset))
		return examples

	def _create_examples_from_dialog(self, dialog, schemas, dataset):
		"""Create examples for every turn in the dialog."""
		dialog_id = dialog["dialogue_id"]
		prev_states = {}
		examples = []
		for turn_idx, turn in enumerate(dialog["turns"]):
			# Generate an example for every frame in every user turn.
			if turn["speaker"] == "USER":
				user_utterance = turn["utterance"]
				user_frames = {f["service"]: f for f in turn["frames"]}
				if turn_idx > 0:
					system_turn = dialog["turns"][turn_idx - 1]
					system_utterance = system_turn["utterance"]
					system_frames = {f["service"]: f for f in system_turn["frames"]}
				else:
					system_utterance = ""
					system_frames = {}
				turn_id = "{}-{}-{:02d}".format(dataset, dialog_id, turn_idx)
				turn_examples, prev_states = self._create_examples_from_turn(
					turn_id, system_utterance, user_utterance, system_frames,
					user_frames, prev_states, schemas)
				examples.extend(turn_examples)
		return examples

	def _get_state_update(self, current_state, prev_state):
		state_update = dict(current_state)
		for slot, values in current_state.items():
			if slot in prev_state and prev_state[slot][0] in values:
				# Remove the slot from state if its value didn't change.
				state_update.pop(slot)
		return state_update

	def _create_examples_from_turn(self, turn_id, system_utterance,
								   user_utterance, system_frames, user_frames,
								   prev_states, schemas):
		"""Creates an example for each frame in the user turn."""
		system_tokens, system_alignments, system_inv_alignments = (
			self._tokenize(system_utterance))
		user_tokens, user_alignments, user_inv_alignments = (
			self._tokenize(user_utterance))
		states = {}
		base_example = InputExample(
			max_seq_length=self._max_seq_length,
			is_real_example=True,
			tokenizer=self._tokenizer,
			log_data_warnings=self._log_data_warnings)
		base_example.example_id = turn_id
		base_example.add_utterance_features(system_tokens, system_inv_alignments,
											user_tokens, user_inv_alignments)
		examples = []
		for service, user_frame in user_frames.items():
			# Create an example for this service.
			example = base_example.make_copy_with_utterance_features()
			example.example_id = "{}-{}".format(turn_id, service)
			example.service_schema = schemas.get_service_schema(service)
			system_frame = system_frames.get(service, None)
			state = user_frame["state"]["slot_values"]
			state_update = self._get_state_update(state, prev_states.get(service, {}))
			states[service] = state
			# Populate features in the example.
			example.add_categorical_slots(state_update)
			# The input tokens to bert are in the format [CLS] [S1] [S2] ... [SEP]
			# [U1] [U2] ... [SEP] [PAD] ... [PAD]. For system token indices a bias of
			# 1 is added for the [CLS] token and for user tokens a bias of 2 +
			# len(system_tokens) is added to account for [CLS], system tokens and
			# [SEP].
			user_span_boundaries = self._find_subword_indices(
				state_update, user_utterance, user_frame["slots"], user_alignments,
				user_tokens, 2 + len(system_tokens))
			if system_frame is not None:
				system_span_boundaries = self._find_subword_indices(
					state_update, system_utterance, system_frame["slots"],
					system_alignments, system_tokens, 1)
			else:
				system_span_boundaries = {}
			example.add_noncategorical_slots(state_update, user_span_boundaries,
											 system_span_boundaries)
			example.add_requested_slots(user_frame)
			example.add_intents(user_frame)
			examples.append(example)
		return examples, states

	def _find_subword_indices(self, slot_values, utterance, char_slot_spans,
							  alignments, subwords, bias):
		"""Find indices for subwords corresponding to slot values."""
		span_boundaries = {}
		for slot, values in slot_values.items():
			# Get all values present in the utterance for the specified slot.
			value_char_spans = {}
			for slot_span in char_slot_spans:
				if slot_span["slot"] == slot:
					value = utterance[slot_span["start"]:slot_span["exclusive_end"]]
					start_tok_idx = alignments[slot_span["start"]]
					end_tok_idx = alignments[slot_span["exclusive_end"] - 1]
					if 0 <= start_tok_idx < len(subwords):
						end_tok_idx = min(end_tok_idx, len(subwords) - 1)
						value_char_spans[value] = (start_tok_idx + bias, end_tok_idx + bias)
			for v in values:
				if v in value_char_spans:
					span_boundaries[slot] = value_char_spans[v]
					break
		return span_boundaries

	def _tokenize(self, utterance):
		"""Tokenize the utterance using word-piece tokenization used by BERT.

		Args:
		  utterance: A string containing the utterance to be tokenized.

		Returns:
		  bert_tokens: A list of tokens obtained by word-piece tokenization of the
			utterance.
		  alignments: A dict mapping indices of characters corresponding to start
			and end positions of words (not subwords) to corresponding indices in
			bert_tokens list.
		  inverse_alignments: A list of size equal to bert_tokens. Each element is a
			tuple containing the index of the starting and inclusive ending
			character of the word corresponding to the subword. This list is used
			during inference to map word-piece indices to spans in the original
			utterance.
		"""
		# TODO: check this out
		# utterance = tokenization.convert_to_unicode(utterance)

		# After _naive_tokenize, spaces and punctuation marks are all retained, i.e.
		# direct concatenation of all the tokens in the sequence will be the
		# original string.
		tokens = _naive_tokenize(utterance)
		# Filter out empty tokens and obtain aligned character index for each token.
		alignments = {}
		char_index = 0
		bert_tokens = []
		# These lists store inverse alignments to be used during inference.
		bert_tokens_start_chars = []
		bert_tokens_end_chars = []
		for token in tokens:
			if token.strip():
				subwords = self._tokenizer.tokenize(token)
				# Store the alignment for the index of starting character and the
				# inclusive ending character of the token.
				alignments[char_index] = len(bert_tokens)
				bert_tokens_start_chars.extend([char_index] * len(subwords))
				bert_tokens.extend(subwords)
				# The inclusive ending character index corresponding to the word.
				inclusive_char_end = char_index + len(token) - 1
				alignments[inclusive_char_end] = len(bert_tokens) - 1
				bert_tokens_end_chars.extend([inclusive_char_end] * len(subwords))
			char_index += len(token)
		inverse_alignments = list(
			zip(bert_tokens_start_chars, bert_tokens_end_chars))
		return bert_tokens, alignments, inverse_alignments

	def get_num_dialog_examples(self, dataset):
		"""Get the number of dilaog examples in the data split.

		Args:
		  dataset: str. can be "train", "dev", or "test".

		Returns:
		  example_count: int. number of examples in the specified dataset.
		"""
		example_count = 0
		dialog_paths = [
			os.path.join(self.dstc8_data_dir, dataset,
						 "dialogues_{:03d}.json".format(i))
			for i in self._file_ranges[dataset]
		]
		dst_set = load_dialogues(dialog_paths)
		for dialog in dst_set:
			for turn in dialog["turns"]:
				if turn["speaker"] == "USER":
					example_count += len(turn["frames"])
		return example_count


class ServiceSchema(object):
	"""A wrapper for schema for a service."""

	def __init__(self, schema_json, service_id=None):
		self._service_name = schema_json["service_name"]
		self._description = schema_json["description"]
		self._schema_json = schema_json
		self._service_id = service_id

		# Construct the vocabulary for intents, slots, categorical slots,
		# non-categorical slots and categorical slot values. These vocabs are used
		# for generating indices for their embedding matrix.
		self._intents = sorted(i["name"] for i in schema_json["intents"])
		self._slots = sorted(s["name"] for s in schema_json["slots"])
		self._categorical_slots = sorted(
			s["name"]
			for s in schema_json["slots"]
			if s["is_categorical"] and s["name"] in self.state_slots)
		self._non_categorical_slots = sorted(
			s["name"]
			for s in schema_json["slots"]
			if not s["is_categorical"] and s["name"] in self.state_slots)
		slot_schemas = {s["name"]: s for s in schema_json["slots"]}
		categorical_slot_values = {}
		categorical_slot_value_ids = {}
		for slot in self._categorical_slots:
			slot_schema = slot_schemas[slot]
			values = sorted(slot_schema["possible_values"])
			categorical_slot_values[slot] = values
			value_ids = {value: idx for idx, value in enumerate(values)}
			categorical_slot_value_ids[slot] = value_ids
		self._categorical_slot_values = categorical_slot_values
		self._categorical_slot_value_ids = categorical_slot_value_ids

	@property
	def schema_json(self):
		return self._schema_json

	@property
	def state_slots(self):
		"""Set of slots which are permitted to be in the dialogue state."""
		state_slots = set()
		for intent in self._schema_json["intents"]:
			state_slots.update(intent["required_slots"])
			state_slots.update(intent["optional_slots"])
		return state_slots

	@property
	def service_name(self):
		return self._service_name

	@property
	def service_id(self):
		return self._service_id

	@property
	def description(self):
		return self._description

	@property
	def slots(self):
		return self._slots

	@property
	def intents(self):
		return self._intents

	@property
	def categorical_slots(self):
		return self._categorical_slots

	@property
	def non_categorical_slots(self):
		return self._non_categorical_slots

	def get_categorical_slot_values(self, slot):
		return self._categorical_slot_values[slot]

	def get_slot_from_id(self, slot_id):
		return self._slots[slot_id]

	def get_intent_from_id(self, intent_id):
		return self._intents[intent_id]

	def get_categorical_slot_from_id(self, slot_id):
		return self._categorical_slots[slot_id]

	def get_non_categorical_slot_from_id(self, slot_id):
		return self._non_categorical_slots[slot_id]

	def get_categorical_slot_value_from_id(self, slot_id, value_id):
		slot = self.categorical_slots[slot_id]
		return self._categorical_slot_values[slot][value_id]

	def get_categorical_slot_value_id(self, slot, value):
		return self._categorical_slot_value_ids[slot][value]


class Schema(object):
	"""Wrapper for schemas for all services in a dataset."""

	def __init__(self, schema_json_path):
		# Load the schema from the json file.
		with open(schema_json_path, "r") as f:
			schemas = json.load(f)
		self._services = sorted(schema["service_name"] for schema in schemas)
		self._services_vocab = {v: k for k, v in enumerate(self._services)}
		service_schemas = {}
		for schema in schemas:
			service = schema["service_name"]
			service_schemas[service] = ServiceSchema(
				schema, service_id=self.get_service_id(service))
		self._service_schemas = service_schemas

	def get_service_id(self, service):
		return self._services_vocab[service]

	def get_service_from_id(self, service_id):
		return self._services[service_id]

	def get_service_schema(self, service):
		return self._service_schemas[service]

	@property
	def services(self):
		return self._services


class InputExample(object):
	"""An example for training/inference."""

	def __init__(self,
				 max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
				 service_schema=None,
				 example_id="NONE",
				 is_real_example=False,
				 tokenizer=None,
				 log_data_warnings=False):
		"""Constructs an InputExample.

		Args:
		  max_seq_length: The maximum length of the sequence. Sequences longer than
			this value will be truncated.
		  service_schema: A ServiceSchema object wrapping the schema for the service
			corresponding to this example.
		  example_id: Unique identifier for the example.
		  is_real_example: Indicates if an example is real or used for padding in a
			minibatch.
		  tokenizer: A tokenizer object that has convert_tokens_to_ids and
			convert_ids_to_tokens methods. It must be non-None when
			is_real_example=True.
		  log_data_warnings: If True, warnings generted while processing data are
			logged. This is useful for debugging data processing.
		"""
		self.service_schema = service_schema
		self.example_id = example_id
		self.is_real_example = is_real_example
		self._max_seq_length = max_seq_length
		self._tokenizer = tokenizer
		self._log_data_warnings = log_data_warnings
		if self.is_real_example and self._tokenizer is None:
			raise ValueError("Must specify tokenizer when input is a real example.")

		# The id of each subword in the vocabulary for BERT.
		self.utterance_ids = [0] * self._max_seq_length
		# Denotes the identity of the sequence. Takes values 0 (system utterance)
		# and 1 (user utterance).
		self.utterance_segment = [0] * self._max_seq_length
		# Mask which takes the value 0 for padded tokens and 1 otherwise.
		self.utterance_mask = [0] * self._max_seq_length
		# Start and inclusive end character indices in the original utterance
		# corresponding to the tokens. This is used to obtain the character indices
		# from the predicted subword indices during inference.
		# NOTE: A positive value indicates the character indices in the user
		# utterance whereas a negative value indicates the character indices in the
		# system utterance. The indices are offset by 1 to prevent ambiguity in the
		# 0 index, which could be in either the user or system utterance by the
		# above convention. Now the 0 index corresponds to padded tokens.
		self.start_char_idx = [0] * self._max_seq_length
		self.end_char_idx = [0] * self._max_seq_length

		# Number of categorical slots present in the service.
		self.num_categorical_slots = 0
		# The status of each categorical slot in the service.
		self.categorical_slot_status = [STATUS_OFF] * MAX_NUM_CAT_SLOT
		# Number of values taken by each categorical slot.
		self.num_categorical_slot_values = [0] * MAX_NUM_CAT_SLOT
		# The index of the correct value for each categorical slot.
		self.categorical_slot_values = [0] * MAX_NUM_CAT_SLOT

		# Number of non-categorical slots present in the service.
		self.num_noncategorical_slots = 0
		# The status of each non-categorical slot in the service.
		self.noncategorical_slot_status = [STATUS_OFF] * MAX_NUM_NONCAT_SLOT
		# The index of the starting subword corresponding to the slot span for a
		# non-categorical slot value.
		self.noncategorical_slot_value_start = [0] * MAX_NUM_NONCAT_SLOT
		# The index of the ending (inclusive) subword corresponding to the slot span
		# for a non-categorical slot value.
		self.noncategorical_slot_value_end = [0] * MAX_NUM_NONCAT_SLOT

		# Total number of slots present in the service. All slots are included here
		# since every slot can be requested.
		self.num_slots = 0
		# Takes value 1 if the corresponding slot is requested, 0 otherwise.
		self.requested_slot_status = [STATUS_OFF] * (
				MAX_NUM_CAT_SLOT + MAX_NUM_NONCAT_SLOT)

		# Total number of intents present in the service.
		self.num_intents = 0
		# Takes value 1 if the intent is active, 0 otherwise.
		self.intent_status = [STATUS_OFF] * MAX_NUM_INTENT

	@property
	def readable_summary(self):
		"""Get a readable dict that summarizes the attributes of an InputExample."""
		seq_length = sum(self.utterance_mask)
		utt_toks = self._tokenizer.convert_ids_to_tokens(
			self.utterance_ids[:seq_length])
		utt_tok_mask_pairs = list(
			zip(utt_toks, self.utterance_segment[:seq_length]))
		active_intents = [
			self.service_schema.get_intent_from_id(idx)
			for idx, s in enumerate(self.intent_status)
			if s == STATUS_ACTIVE
		]
		if len(active_intents) > 1:
			raise ValueError(
				"Should not have multiple active intents in a single service.")
		active_intent = active_intents[0] if active_intents else ""
		slot_values_in_state = {}
		for idx, s in enumerate(self.categorical_slot_status):
			if s == STATUS_ACTIVE:
				value_id = self.categorical_slot_values[idx]
				slot_values_in_state[self.service_schema.get_categorical_slot_from_id(
					idx)] = self.service_schema.get_categorical_slot_value_from_id(
					idx, value_id)
			elif s == STATUS_DONTCARE:
				slot_values_in_state[self.service_schema.get_categorical_slot_from_id(
					idx)] = STR_DONTCARE
		for idx, s in enumerate(self.noncategorical_slot_status):
			if s == STATUS_ACTIVE:
				slot = self.service_schema.get_non_categorical_slot_from_id(idx)
				start_id = self.noncategorical_slot_value_start[idx]
				end_id = self.noncategorical_slot_value_end[idx]
				# Token list is consisted of the subwords that may start with "##". We
				# remove "##" to reconstruct the original value. Note that it's not a
				# strict restoration of the original string. It's primarily used for
				# debugging.
				# ex. ["san", "j", "##ose"] --> "san jose"
				readable_value = " ".join(utt_toks[start_id:end_id + 1]).replace(
					" ##", "")
				slot_values_in_state[slot] = readable_value
			elif s == STATUS_DONTCARE:
				slot = self.service_schema.get_non_categorical_slot_from_id(idx)
				slot_values_in_state[slot] = STR_DONTCARE

		summary_dict = {
			"utt_tok_mask_pairs": utt_tok_mask_pairs,
			"utt_len": seq_length,
			"num_categorical_slots": self.num_categorical_slots,
			"num_categorical_slot_values": self.num_categorical_slot_values,
			"num_noncategorical_slots": self.num_noncategorical_slots,
			"service_name": self.service_schema.service_name,
			"active_intent": active_intent,
			"slot_values_in_state": slot_values_in_state
		}
		return summary_dict

	def add_utterance_features(self, system_tokens, system_inv_alignments,
							   user_tokens, user_inv_alignments):
		"""Add utterance related features input to bert.

		Note: this method modifies the system tokens and user_tokens in place to
		make their total length <= the maximum input length for BERT model.

		Args:
		  system_tokens: a list of strings which represents system utterance.
		  system_inv_alignments: a list of tuples which denotes the start and end
			charater of the tpken that a bert token originates from in the original
			system utterance.
		  user_tokens: a list of strings which represents user utterance.
		  user_inv_alignments: a list of tuples which denotes the start and end
			charater of the token that a bert token originates from in the original
			user utterance.
		"""
		# Make user-system utterance input (in BERT format)
		# Input sequence length for utterance BERT encoder
		max_utt_len = self._max_seq_length

		# Modify lengths of sys & usr utterance so that length of total utt
		# (including [CLS], [SEP], [SEP]) is no more than max_utt_len
		is_too_long = truncate_seq_pair(system_tokens, user_tokens, max_utt_len - 3)
		if is_too_long and self._log_data_warnings:
			logger.info(f'Utterance sequence truncated in example id - {self.example_id}.')

		# Construct the tokens, segment mask and valid token mask which will be
		# input to BERT, using the tokens for system utterance (sequence A) and
		# user utterance (sequence B).
		utt_subword = []
		utt_seg = []
		utt_mask = []
		start_char_idx = []
		end_char_idx = []

		utt_subword.append("[CLS]")
		utt_seg.append(0)
		utt_mask.append(1)
		start_char_idx.append(0)
		end_char_idx.append(0)

		for subword_idx, subword in enumerate(system_tokens):
			utt_subword.append(subword)
			utt_seg.append(0)
			utt_mask.append(1)
			st, en = system_inv_alignments[subword_idx]
			start_char_idx.append(-(st + 1))
			end_char_idx.append(-(en + 1))

		utt_subword.append("[SEP]")
		utt_seg.append(0)
		utt_mask.append(1)
		start_char_idx.append(0)
		end_char_idx.append(0)

		for subword_idx, subword in enumerate(user_tokens):
			utt_subword.append(subword)
			utt_seg.append(1)
			utt_mask.append(1)
			st, en = user_inv_alignments[subword_idx]
			start_char_idx.append(st + 1)
			end_char_idx.append(en + 1)

		utt_subword.append("[SEP]")
		utt_seg.append(1)
		utt_mask.append(1)
		start_char_idx.append(0)
		end_char_idx.append(0)

		utterance_ids = self._tokenizer.convert_tokens_to_ids(utt_subword)

		# Zero-pad up to the BERT input sequence length.
		while len(utterance_ids) < max_utt_len:
			utterance_ids.append(0)
			utt_seg.append(0)
			utt_mask.append(0)
			start_char_idx.append(0)
			end_char_idx.append(0)
		self.utterance_ids = utterance_ids
		self.utterance_segment = utt_seg
		self.utterance_mask = utt_mask
		self.start_char_idx = start_char_idx
		self.end_char_idx = end_char_idx

	def make_copy_with_utterance_features(self):
		"""Make a copy of the current example with utterance features."""
		new_example = InputExample(
			max_seq_length=self._max_seq_length,
			service_schema=self.service_schema,
			example_id=self.example_id,
			is_real_example=self.is_real_example,
			tokenizer=self._tokenizer,
			log_data_warnings=self._log_data_warnings)
		new_example.utterance_ids = list(self.utterance_ids)
		new_example.utterance_segment = list(self.utterance_segment)
		new_example.utterance_mask = list(self.utterance_mask)
		new_example.start_char_idx = list(self.start_char_idx)
		new_example.end_char_idx = list(self.end_char_idx)
		return new_example

	def add_categorical_slots(self, state_update):
		"""Add features for categorical slots."""
		categorical_slots = self.service_schema.categorical_slots
		self.num_categorical_slots = len(categorical_slots)
		for slot_idx, slot in enumerate(categorical_slots):
			values = state_update.get(slot, [])
			# Add categorical slot value features.
			slot_values = self.service_schema.get_categorical_slot_values(slot)
			self.num_categorical_slot_values[slot_idx] = len(slot_values)
			if not values:
				self.categorical_slot_status[slot_idx] = STATUS_OFF
			elif values[0] == STR_DONTCARE:
				self.categorical_slot_status[slot_idx] = STATUS_DONTCARE
			else:
				self.categorical_slot_status[slot_idx] = STATUS_ACTIVE
				self.categorical_slot_values[slot_idx] = (
					self.service_schema.get_categorical_slot_value_id(slot, values[0]))

	def add_noncategorical_slots(self, state_update, system_span_boundaries,
								 user_span_boundaries):
		"""Add features for non-categorical slots."""
		noncategorical_slots = self.service_schema.non_categorical_slots
		self.num_noncategorical_slots = len(noncategorical_slots)
		for slot_idx, slot in enumerate(noncategorical_slots):
			values = state_update.get(slot, [])
			if not values:
				self.noncategorical_slot_status[slot_idx] = STATUS_OFF
			elif values[0] == STR_DONTCARE:
				self.noncategorical_slot_status[slot_idx] = STATUS_DONTCARE
			else:
				self.noncategorical_slot_status[slot_idx] = STATUS_ACTIVE
				# Add indices of the start and end tokens for the first encountered
				# value. Spans in user utterance are prioritized over the system
				# utterance. If a span is not found, the slot value is ignored.
				if slot in user_span_boundaries:
					start, end = user_span_boundaries[slot]
				elif slot in system_span_boundaries:
					start, end = system_span_boundaries[slot]
				else:
					# A span may not be found because the value was cropped out or because
					# the value was mentioned earlier in the dialogue. Since this model
					# only makes use of the last two utterances to predict state updates,
					# it will fail in such cases.
					if self._log_data_warnings:
						logger.info(f'"Slot values {str(values)} not found in user or system utterance in example with id - {self.example_id}.')

					continue
				self.noncategorical_slot_value_start[slot_idx] = start
				self.noncategorical_slot_value_end[slot_idx] = end

	def add_requested_slots(self, frame):
		all_slots = self.service_schema.slots
		self.num_slots = len(all_slots)
		for slot_idx, slot in enumerate(all_slots):
			if slot in frame["state"]["requested_slots"]:
				self.requested_slot_status[slot_idx] = STATUS_ACTIVE

	def add_intents(self, frame):
		all_intents = self.service_schema.intents
		self.num_intents = len(all_intents)
		for intent_idx, intent in enumerate(all_intents):
			if intent == frame["state"]["active_intent"]:
				self.intent_status[intent_idx] = STATUS_ACTIVE


# Modified from run_classifier._truncate_seq_pair in the public bert model repo.
# https://github.com/google-research/bert/blob/master/run_classifier.py.
def truncate_seq_pair(tokens_a, tokens_b, max_length):
	"""Truncate a seq pair in place so that their total length <= max_length."""
	is_too_long = False
	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		is_too_long = True
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()
	return is_too_long


def _naive_tokenize(s):
	"""Tokenize a string, separating words, spaces and punctuations."""
	# Spaces and punctuation marks are all retained, i.e. direct concatenation
	# of all the tokens in the sequence will be the original string.
	seq_tok = [tok for tok in re.split(r"([^a-zA-Z0-9])", s) if tok]
	return seq_tok


def process_sgd(infold, uncased, dataset_name, task_name, tokenizer, encoder,
				dataset_split, max_seq_length, modes, log_data_warnings):
	""" process and convert SGD dataset into CSV files
	"""
	outfold = f'{infold}/{dataset_name}-nemo-processed'
	infold = f'{infold}/'

	if uncased:
		outfold = f'{outfold}-uncased'

	# if if_exist(outfold, ['dialogues.tsv']):
	# 	logger.info(DATABASE_EXISTS_TMP.format(dataset_name, outfold))
	# 	return outfold

	logger.info(f'Processing {dataset_name} dataset and store at {outfold}')

	os.makedirs(outfold, exist_ok=True)

	processor = Dstc8DataProcessor(
		infold,
		train_file_range=FILE_RANGES[task_name]["train"],
		dev_file_range=FILE_RANGES[task_name]["dev"],
		test_file_range=FILE_RANGES[task_name]["test"],
		tokenizer=tokenizer,
		max_seq_length=max_seq_length,
		log_data_warnings=log_data_warnings)

	logger.info("Start generating the dialogue examples.")
	_create_dialog_examples(processor, outfold + "/dialogues.tsv", dataset_split)
	logger.info("Finish generating the dialogue examples.")

	# Generate the schema embeddings if needed or specified.
	# vocab_file = os.path.join(FLAGS.bert_ckpt_dir, "vocab.txt")
	# bert_init_ckpt = os.path.join(FLAGS.bert_ckpt_dir, "bert_model.ckpt")
	# tokenization.validate_case_matches_checkpoint(
	# 	do_lower_case=FLAGS.do_lower_case, init_checkpoint=bert_init_ckpt)
	#
	# bert_config = modeling.BertConfig.from_json_file(
	# 	os.path.join(FLAGS.bert_ckpt_dir, "bert_config.json"))
	# if FLAGS.max_seq_length > bert_config.max_position_embeddings:
	# 	raise ValueError(
	# 		"Cannot use sequence length %d because the BERT model "
	# 		"was only trained up to sequence length %d" %
	# 		(FLAGS.max_seq_length, bert_config.max_position_embeddings))

	schema_embedding_file = os.path.join(
		infold,
		f"{dataset_split}_pretrained_schema_embedding.npy")
	if not if_exist("", [schema_embedding_file]):
		logger.info("Start generating the schema embeddings.")
		_create_schema_embeddings(encoder, schema_embedding_file)
		logger.info("Finish generating the schema embeddings.")

	return outfold


def _create_schema_embeddings(encoder, schema_embedding_file):
	"""Create schema embeddings and save it into file."""
	# if not tf.io.gfile.exists(FLAGS.schema_embedding_dir):
	# 	tf.io.gfile.makedirs(FLAGS.schema_embedding_dir)

	schema_emb_run_config = tf.contrib.tpu.RunConfig(
		master=FLAGS.master,
		tpu_config=tf.contrib.tpu.TPUConfig(
			num_shards=FLAGS.num_tpu_cores,
			per_host_input_for_training=is_per_host))

	schema_json_path = os.path.join(FLAGS.dstc8_data_dir, FLAGS.dataset_split,
									"schema.json")
	schemas = schema.Schema(schema_json_path)

	# Prepare BERT model for embedding a natural language descriptions.
	bert_init_ckpt = os.path.join(FLAGS.bert_ckpt_dir, "bert_model.ckpt")
	schema_emb_model_fn = extract_schema_embedding.model_fn_builder(
		bert_config=bert_config,
		init_checkpoint=bert_init_ckpt,
		use_tpu=FLAGS.use_tpu,
		use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)
	# If TPU is not available, this will fall back to normal Estimator on CPU
	# or GPU.
	schema_emb_estimator = tf.contrib.tpu.TPUEstimator(
		use_tpu=FLAGS.use_tpu,
		model_fn=schema_emb_model_fn,
		config=schema_emb_run_config,
		predict_batch_size=FLAGS.predict_batch_size)
	vocab_file = os.path.join(FLAGS.bert_ckpt_dir, "vocab.txt")
	tokenizer = tokenization.FullTokenizer(
		vocab_file=vocab_file, do_lower_case=FLAGS.do_lower_case)
	emb_generator = extract_schema_embedding.SchemaEmbeddingGenerator(
		tokenizer, schema_emb_estimator, FLAGS.max_seq_length)
	emb_generator.save_embeddings(schemas, schema_embedding_file)


def load_dialogues(dialog_json_filepaths):
	"""Obtain the list of all dialogues from specified json files."""
	dialogs = []
	for dialog_json_filepath in sorted(dialog_json_filepaths):
		with open(dialog_json_filepath, 'r') as f:
			dialogs.extend(json.load(f))
	return dialogs


def _create_dialog_examples(processor, dial_file, dataset_split):
	"""Create dialog examples and save in the file."""
	frame_examples = processor.get_dialog_examples(dataset_split)
	file_based_convert_examples_to_features(frame_examples, dial_file)


def list_to_str(l):
	return " ".join(str(x) for x in l)


# Modified from run_classifier.file_based_convert_examples_to_features in the
# public bert model repo.
# https://github.com/google-research/bert/blob/master/run_classifier.py.
def file_based_convert_examples_to_features(dial_examples, output_file):
	"""Convert a set of `InputExample`s to a TFRecord file."""
	# Changed Here
	# writer = tf.io.TFRecordWriter(output_file)
	writer = open(output_file, "w")

	for (ex_index, example) in enumerate(dial_examples):
		if ex_index % 10000 == 0:
			logger.info(f'Writing example {ex_index} of {len(dial_examples)}')

		# if isinstance(example, PaddingInputExample):
		# 	ex = InputExample()
		# else:
		# 	ex = example

		ex = example
		features = collections.OrderedDict()

		features["example_id"] = ex.example_id
		features["is_real_example"] = str(int(ex.is_real_example))
		features["service_id"] = str(ex.service_schema.service_id)

		features["utt"] = list_to_str(ex.utterance_ids)
		features["utt_seg"] = list_to_str(ex.utterance_segment)
		features["utt_mask"] = list_to_str(ex.utterance_mask)

		features["cat_slot_num"] = str(ex.num_categorical_slots)
		features["cat_slot_status"] = list_to_str(ex.categorical_slot_status)
		features["cat_slot_value_num"] = list_to_str(ex.num_categorical_slot_values)
		features["cat_slot_value"] = list_to_str(ex.categorical_slot_values)

		features["noncat_slot_num"] = str(ex.num_noncategorical_slots)
		features["noncat_slot_status"] = list_to_str(ex.noncategorical_slot_status)
		features["noncat_slot_value_start"] = list_to_str(ex.noncategorical_slot_value_start)
		features["noncat_slot_value_end"] = list_to_str(ex.noncategorical_slot_value_end)
		features["noncat_alignment_start"] = list_to_str(ex.start_char_idx)
		features["noncat_alignment_end"] = list_to_str(ex.end_char_idx)

		features["req_slot_num"] = str(ex.num_slots)
		features["req_slot_status"] = list_to_str(ex.requested_slot_status)

		features["intent_num"] = str(ex.num_intents)
		features["intent_status"] = list_to_str(ex.intent_status)

		if ex_index == 0:
			header = "\t".join(features.keys())
			writer.write(header + "\n")
		# tf_example = tf.train.Example(features=tf.train.Features(feature=features))
		csv_example = "\t".join(list(features.values()))
		writer.write(csv_example + "\n")
	writer.close()

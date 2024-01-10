import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_new_inputs(output_start, output_end, utterances, top_k):
    # Extract top_k spans with constraint that start index must be greater than end index
    span_extraction = []
    inputs = []
    for k, sample_o_s in enumerate(output_start):
        tmp_span_extraction = []
        for i, score in enumerate(sample_o_s):
            for j in range(i + 1, len(output_end[k])):
                sum_score = score.item() + output_end[k][j].item()
                sum_score = sum_score / (j - i)
                tmp_span_extraction.append((i, j, sum_score))
        best_spans = sorted(tmp_span_extraction, key=lambda t: t[2], reverse=True)[
            :top_k
        ]
        span_extraction.append(best_spans)

    inputs = []
    for i, sample in enumerate(span_extraction):
        tmp_input = []
        for span in sample:
            tmp_input.append(utterances[i][span[0] : span[1]])
        inputs.append(tmp_input)
    # Inputs is a list of lists (top_k) of tensors of token ids (span_len)

    # We should pad them (assumption top_k is 1)
    inputs = [input[0] for input in inputs]
    lengths = [len(seq) for seq in inputs]
    max_len = 1 if max(lengths) == 0 else max(lengths)
    attention_masks = torch.FloatTensor(
        [
            [1 for i in range(len(seq))] + [0 for i in range(max_len - len(seq))]
            for seq in inputs
        ]
    )
    # Pad token is zero in our case
    # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
    # batch_size X maximum length of a sequence
    padded_inputs = torch.LongTensor(len(inputs), max_len).fill_(0)
    for i, seq in enumerate(inputs):
        end = lengths[i]
        padded_inputs[i, :end] = seq  # We copy each sequence into the matrix
    # print(padded_seqs)
    padded_inputs = (
        padded_inputs.detach()
    )  # We remove these tensors from the computational graph
    return padded_inputs, attention_masks


def span_train_loop(
    train_loader, model, span_criterion, pol_criterion, optimizer, top_k
):
    model.train()
    best_loss = 0
    overall_loss = []
    for sample in train_loader:
        optimizer.zero_grad()
        masks = sample["utt_mask"]
        utterances = sample["utterance"]
        one_hots = sample["one_hot"]
        # print(sample['utterance'].size(), sample['utt_mask'].size(), sample['utt_len'].size(), sample['start'].size(), sample['end'].size())
        output_start, output_end, soft_output_start, soft_output_end = model(
            utterances, masks, mode="span"
        )
        span_loss = span_criterion(
            soft_output_start, soft_output_end, sample["start"], sample["end"]
        )

        padded_inputs, attention_masks = compute_new_inputs(
            output_start, output_end, utterances, top_k
        )

        padded_inputs = padded_inputs.to(device)
        attention_masks = attention_masks.to(device)
        # Input to the model
        output, soft_output = model(padded_inputs, attention_masks, mode="polarity")
        # Compute loss
        pol_loss = pol_criterion(soft_output, one_hots)
        # Sum losses
        loss = span_loss + pol_loss
        overall_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    overall_loss_item = sum(overall_loss) / len(overall_loss)
    return overall_loss_item


def span_dev_loop(
    SPAN_PATH, dev_loader, model, span_criterion, pol_criterion, best_span_loss, top_k
):
    model.eval()
    overall_loss = []
    for sample in dev_loader:
        masks = sample["utt_mask"]
        utterances = sample["utterance"]
        one_hots = sample["one_hot"]
        # print(sample['utterance'].size(), sample['utt_mask'].size(), sample['utt_len'].size(), sample['start'].size(), sample['end'].size())
        output_start, output_end, soft_output_start, soft_output_end = model(
            utterances, masks, mode="span"
        )
        span_loss = span_criterion(
            soft_output_start, soft_output_end, sample["start"], sample["end"]
        )

        padded_inputs, attention_masks = compute_new_inputs(
            output_start, output_end, utterances, top_k
        )

        padded_inputs = padded_inputs.to(device)
        attention_masks = attention_masks.to(device)
        # Input to the model
        output, soft_output = model(padded_inputs, attention_masks, mode="polarity")
        # Compute loss
        pol_loss = pol_criterion(soft_output, one_hots)
        # Sum losses
        loss = span_loss + pol_loss

        overall_loss.append(loss.item())

    overall_loss_item = sum(overall_loss) / len(overall_loss)
    if overall_loss_item < best_span_loss:
        best_span_loss = overall_loss_item
        torch.save(model.state_dict(), SPAN_PATH)
    return overall_loss_item, best_span_loss


def span_test_loop(test_loader, model, span_criterion, pol_criterion, top_k):
    model.eval()
    best_loss = 0
    overall_loss = []
    for sample in test_loader:
        masks = sample["utt_mask"]
        utterances = sample["utterance"]
        one_hots = sample["one_hot"]
        # print(sample['utterance'].size(), sample['utt_mask'].size(), sample['utt_len'].size(), sample['start'].size(), sample['end'].size())
        output_start, output_end, soft_output_start, soft_output_end = model(
            utterances, masks, mode="span"
        )
        span_loss = span_criterion(
            soft_output_start, soft_output_end, sample["start"], sample["end"]
        )

        padded_inputs, attention_masks = compute_new_inputs(
            output_start, output_end, utterances, top_k
        )

        padded_inputs = padded_inputs.to(device)
        attention_masks = attention_masks.to(device)
        # Input to the model
        output, soft_output = model(padded_inputs, attention_masks, mode="polarity")
        # Compute loss
        pol_loss = pol_criterion(soft_output, one_hots)
        # Sum losses
        loss = span_loss + pol_loss

        overall_loss.append(loss.item())
    overall_loss_item = sum(overall_loss) / len(overall_loss)
    print(f"Loss {overall_loss}")
    return overall_loss_item

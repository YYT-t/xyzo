from torch.nn import CrossEntropyLoss
import torch
def regularized_logp(logits, labels, vocab_size, label_smoothing, reduction):
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = CrossEntropyLoss(label_smoothing=label_smoothing, reduction=reduction)
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss

def regularized_logp_tensor(logits, labels, vocab_size, label_smoothing):
    logits = logits.float()
    # print("logits.shape=", logits.shape)
    # print("labels.shape=", labels.shape)

    loss_fct = CrossEntropyLoss(label_smoothing=label_smoothing)

    tensor_loss = torch.zeros(logits.shape[0]).to(logits.device)
    for i in range(logits.shape[0]):
        tmp_shift_logits = logits[i, :-1, :]
        tmp_shift_labels = labels[i, 1:]
        tmp_loss = loss_fct(tmp_shift_logits, tmp_shift_labels)
        tensor_loss[i] = tmp_loss
    # print("tensor_loss=", tensor_loss)
    # print("torch.meam(tensor_loss)=", torch.mean(tensor_loss))

    return tensor_loss
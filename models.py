from transformers import PreTrainedModel

class AC_Model(PreTrainedModel):
    def __init__(self, config, actor, critic, ref):
        super(AC_Model, self).__init__(config)
        self.actor = actor
        self.critic = critic
        self.ref = ref
    def actor_forward(self, **kwargs):
        return self.actor(**kwargs)
    def critic_forward(self, **kwargs):
        return self.critic(**kwargs)
    def ref_forward(self, **kwargs):
        return self.ref(**kwargs)

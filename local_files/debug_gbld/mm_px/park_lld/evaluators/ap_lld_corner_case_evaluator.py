from .ap_lld_evaluator import AP_LLDEvaluator

class APLLDGeneralFrontEvaluator(AP_LLDEvaluator):
    def evaluate(self, gt_dict, pred_dict, uuid='', **kwargs):
        self._reset()
        self.process_once(pred_dict, gt_dict, uuid)
        if uuid in self.AutoQA["line_exist"] or uuid in self.AutoQA["line_point"]:
            result = False
        else:
            result = True
        print('general case ', uuid, result)
        return {'aplld_front': {'result': result}}

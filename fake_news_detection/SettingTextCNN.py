from setting import setting


class SettingTextCNN(setting):

    def load_run_save_evaluate(self):
        # load dataset
        dataset, article_train_test_divide_dict, article_credibility_dic = self.dataset.load()

        # run MethodModule
        self.method.data = dataset
        self.method.train_test_divide = article_train_test_divide_dict
        self.method.article_credibility_dict = article_credibility_dic

        textCNN_result = self.method.run()

        self.result.data = textCNN_result
        self.result.save()

        # evaluating ResultModule (optional)
        # evaluation can be done independently after getting the learning ResultModule
        #        self.evaluate.data = learned_result
        return textCNN_result


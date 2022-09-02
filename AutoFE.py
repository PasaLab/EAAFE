import logging

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")


class AutoFE:
    def __init__(self, config=None, X=None, y=None, schema=None):
        self.config = config.config
        self.config['X'] = X
        self.config['y'] = y
        self.config['schema'] = schema

    def run(self):

        logger.info('start run')
        if self.config['search_method'] == "EAAFE":
            from EAAFE.EAAFE import SEARCH_FE_PIPELINE as EAAFE_search
            logger.info('EAAFE')
            self.autofe = EAAFE_search(config=self.config)
            self.autofe.run()
        elif self.config['search_method'] == "EAAFE_SPARK":
            from EAAFE.EAAFE_SPARK import SEARCH_FE_PIPELINE as eaafe_search_spark
            self.autofe = eaafe_search_spark(config=self.config)
            self.autofe.run()
        elif self.config['search_method'] == "EAAFE_RAY":
            from EAAFE.EAAFE_RAY import SEARCH_FE_PIPELINE as eaafe_search_ray
            self.autofe = eaafe_search_ray(config=self.config)
            self.autofe.run()
        else:
            pass

    def transform_sequence(self):
        logger.info('start transform sequence')
        transformed_data = None
        if self.config['search_method'] == "EAAFE":
            from EAAFE.EAAFE import SEARCH_FE_PIPELINE as EAAFE_search
            logger.info('EAAFE')
            self.autofe = EAAFE_search(config=self.config)
            transformed_data = self.autofe.transform_sequence()
        elif self.config['search_method'] == "EAAFE_SPARK":
            from EAAFE.EAAFE_SPARK import SEARCH_FE_PIPELINE as eaafe_search_spark
            self.autofe = eaafe_search_spark(config=self.config)
            transformed_data = self.autofe.transform_sequence()
        elif self.config['search_method'] == "EAAFE_RAY":
            from EAAFE.EAAFE_RAY import SEARCH_FE_PIPELINE as eaafe_search_ray
            self.autofe = eaafe_search_ray(config=self.config)
            transformed_data = self.autofe.transform_sequence()
        else:
            pass
        return transformed_data

    def save(self):
        logger.info('save')

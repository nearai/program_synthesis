from .parser_for_synthesis import KarelForSynthesisParser
from .utils import str2bool, makedirs, pprint, beautify, TimeoutError, KarelSyntaxError

from datetime import datetime

__copyright__ = 'Copyright 2015 - {} by Taehoon Kim'.format(datetime.now().year)
__version__ = '1.3.0'
__license__ = 'BSD'
__author__ = 'Taehoon Kim'
__email__ = 'carpedm20@gmail.com'
__source__ = 'https://github.com/carpedm20/karel'
__description__ = 'Karel dataset for program synthesis and program induction'

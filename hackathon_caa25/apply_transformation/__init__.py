from .altitude import format_altitude
from .binarization import one_hot_encode
from .departements import format_zone
from .derogations import format_derogations
from .financials import format_revenues
from .fires import add_incendies_info
from .houses import format_housing, format_menages
from .individuals import format_individuals
from .temperatures import cross_temperature_data

from .utils_transformation.categorical_converter import categorical_conversion

from d810.optimizers.flow.flattening.unflattener import Unflattener
from d810.optimizers.flow.flattening.unflattener_switch_case import UnflattenerSwitchCase
from d810.optimizers.flow.flattening.unflattener_indirect import UnflattenerTigressIndirect
from d810.optimizers.flow.flattening.unflattener_fake_jump import UnflattenerFakeJump

UNFLATTENING_BLK_RULES = [Unflattener(), UnflattenerSwitchCase(), UnflattenerTigressIndirect(), UnflattenerFakeJump()]

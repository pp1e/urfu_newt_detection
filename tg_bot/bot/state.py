from aiogram.fsm.state import StatesGroup, State


class AddNewtBellyFlow(StatesGroup):
    waiting_for_confirmation = State()
    waiting_for_new_newt_id = State()
    waiting_for_existing_newt_id = State()

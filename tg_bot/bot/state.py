from aiogram.fsm.state import StatesGroup, State


class AddNewtBellyFlow(StatesGroup):
    waiting_for_newt_class_choice = State()
    waiting_for_new_newt_id = State()

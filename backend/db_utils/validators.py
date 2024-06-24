from cerberus import Validator

#スキーマ定義
user_schema = {
    'name': {
        'type': 'string',
        'maxlength': 255,
        'empty': False,
        'nullable': False,
    },
    'password': {
        'type': 'string',
        'maxlength': 255,
        'empty': False,
        'nullable': False,
    },
    'email': {
        'type': 'string',
        'maxlength': 255,
        'empty': False,
        'nullable': False
    },
    'authority': {
        'type': 'integer',
        'nullable': False,
    }
}

#スキーマを選択してデータに対してバリデーションを行う
def validate_data(target,type):
    target_dict = target.dict()
    v = Validator()
    if(type=='user'):
        target_schema = user_schema
    else:
        target_schema = None
    
    return v.validate(target_dict,target_schema)

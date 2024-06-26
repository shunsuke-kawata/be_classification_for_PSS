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
        'type': 'boolean',
        'nullable': False,
    }
}
login_user_schema = {
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
    }
}
project_schema = {
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
    'description': {
        'type': 'string',
        'maxlength': 255,
        'empty': True,
        'nullable': True,
    },
    'owner_id': {
        'type': 'integer',
        'nullable': False,
    }
}
project_membership_schema = {
    'user_id': {
        'type': 'integer',
        'nullable': False,
    },
    'project_id': {
        'type': 'integer',
        'nullable': False,
    }
}
join_user_schema = {
    'user_id': {
        'type': 'integer',
        'nullable': False,
    },
    'project_id': {
        'type': 'integer',
        'nullable': False,
    },
    'project_password': {
        'type': 'string',
        'maxlength': 255,
        'empty': False,
        'nullable': False,
    },
}

#スキーマを選択してデータに対してバリデーションを行う
def validate_data(target,type):
    target_dict = target.dict()
    v = Validator()
    if(type=='user'):
        target_schema = user_schema
    elif(type=='project'):
        target_schema = project_schema
    elif(type=='login_user'):
        target_schema= login_user_schema
    elif(type=='project_membership'):
        target_schema= project_membership_schema
    elif(type=='join_user'):
        target_schema= join_user_schema
    else:
        target_schema = None
    
    if(target_schema is None):
        return False
    return v.validate(target_dict,target_schema)

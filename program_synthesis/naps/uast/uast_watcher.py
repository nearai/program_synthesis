# Watcher is a way to monitor Executor activity
# Example cases:
#  - collect execution traces
#  - collect information about data flowing through statements to later find blocks
#    with the same behaviour across different solutions to the same problem
#
# The current set of events is tailored towards the use cases above.


def tuplify(x):
    if isinstance(x, list):
        return tuple([tuplify(_) for _ in x])
    elif isinstance(x, dict):
        return tuple(x.items())
    return x


class WatcherEvent(object):
    def __init__(self, event_type, executor, context, *args):
        super(WatcherEvent, self).__init__()
        self.context = context
        self.executor = executor
        self.event_type = event_type
        self.args = args


class Watcher(object):
    def __init__(self):
        super(Watcher, self).__init__()


    def watch(self, event):

        if event.event_type == 'before_expression':
            self.on_before_expression(event.executor, event.context, *event.args)
        elif event.event_type == 'after_expression':
            self.on_after_expression(event.executor, event.context, *event.args)
        elif event.event_type == 'before_ternary_expression':
            self.on_before_ternary_expression(event.executor, event.context, *event.args)
        elif event.event_type == 'after_ternary_expression':
            self.on_after_ternary_expression(event.executor, event.context, *event.args)
        elif event.event_type == 'before_statement':
            self.on_before_statement(event.executor, event.context, *event.args)
        elif event.event_type == 'after_statement':
            self.on_after_statement(event.executor, event.context, *event.args)
        elif event.event_type == 'before_block':
            self.on_before_block(event.executor, event.context, *event.args)
        elif event.event_type == 'after_block':
            self.on_after_block(event.executor, event.context, *event.args)
        elif event.event_type == 'before_if_block':
            self.on_before_if_block(event.executor, event.context, *event.args)
        elif event.event_type == 'after_if_block':
            self.on_after_if_block(event.executor, event.context, *event.args)
        elif event.event_type == 'before_foreach_block':
            self.on_before_foreach_block(event.executor, event.context, *event.args)
        elif event.event_type == 'after_foreach_block':
            self.on_after_foreach_block(event.executor, event.context, *event.args)
        elif event.event_type == 'before_while_block':
            self.on_before_while_block(event.executor, event.context, *event.args)
        elif event.event_type == 'after_while_block':
            self.on_after_while_block(event.executor, event.context, *event.args)
        elif event.event_type == 'before_func_block':
            self.on_before_func_block(event.executor, event.context, *event.args)
        elif event.event_type == 'after_func_block':
            self.on_after_func_block(event.executor, event.context, *event.args)
        elif event.event_type == 'before_func':
            self.on_before_func(event.executor, event.context, *event.args)
        elif event.event_type == 'after_func':
            self.on_after_func(event.executor, event.context, *event.args)
        elif event.event_type == 'read':
            self.on_read(event.executor, event.context, *event.args)
        elif event.event_type == 'write':
            self.on_write(event.executor, event.context, *event.args)


    def on_before_expression(self, executor, context, expr, is_lhs=False):
        pass


    def on_after_expression(self, executor, context, ret, expr, is_lhs=False):
        pass


    def on_before_ternary_expression(self, executor, context, pred_expr, expr, is_lhs=False):
        pass


    def on_after_ternary_expression(self, executor, context, ret, pred_expr, expr, is_lhs=False):
        pass


    def on_before_statement(self, executor, context, stmt):
        pass


    def on_after_statement(self, executor, context, ret, stmt):
        pass


    def on_before_block(self, executor, context, block):
        pass


    def on_after_block(self, executor, context, ret, block):
        pass


    def on_before_if_block(self, executor, context, expr, block):
        pass


    def on_after_if_block(self, executor, context, ret, expr, block):
        pass


    def on_before_foreach_block(self, executor, context, expr, block):
        pass


    def on_after_foreach_block(self, executor, context, ret, expr, block):
        pass


    def on_before_while_block(self, executor, context, expr, block):
        pass


    def on_after_while_block(self, executor, context, ret, expr, block):
        pass


    def on_before_func_block(self, executor, context, func_name, func_vars, func_args, args_vals, expressions, block):
        pass


    def on_after_func_block(self, executor, context, ret, func_name, func_vars, func_args, args_vals, expressions, block):
        pass


    def on_before_func(self, executor, context, func, args):
        pass


    def on_after_func(self, executor, context, ret, func, args):
        pass

 
    def on_read(self, executor, context, args):
        pass

 
    def on_write(self, executor, context, args):
        pass


# Copyright (c) 2022 Zhipu.AI

import sys

if sys.argv[1] == 'block':
    from test.test_block import main
    main()
elif sys.argv[1] == 'rel_shift':
    from test.test_rel_shift import main
    main()

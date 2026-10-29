"""
Keep a free task-credit balance per account and gate bids on it (`PR_P4`).

A bid is affordable when the balance covers `n_tasks * p_max`; the balance is
debited only when the bid is matched into a contract.

Import as:

import research.Noesis.market.credit_ledger as rnmacrle
"""

import abc
import logging
from typing import Dict, Optional

import helpers.hdbg as hdbg
import helpers.hprint as hprint
import research.Noesis.market.batch_call_auction as rnbacaau

_LOG = logging.getLogger(__name__)


# #############################################################################
# Constants
# #############################################################################


# Free grant per account, in the unit of `Bid.p_max`.
DEFAULT_FREE_CREDIT = 1.0


# #############################################################################
# CreditLedgerStore
# #############################################################################


class CreditLedgerStore(abc.ABC):
    """
    Pluggable storage backend for `CreditLedger`'s balances and debits.
    """

    @abc.abstractmethod
    def get_balance(self, account_id: str) -> Optional[float]:
        """
        :return: the balance of `account_id`, or `None` if it has none yet
        """
        ...

    @abc.abstractmethod
    def set_balance(self, account_id: str, balance: float) -> None: ...

    @abc.abstractmethod
    def has_debit(self, contract_id: int) -> bool:
        """
        :return: whether `contract_id` was already debited
        """
        ...

    @abc.abstractmethod
    def record_debit(
        self, contract_id: int, account_id: str, amount: float
    ) -> None: ...


# #############################################################################
# _InMemoryCreditLedgerStore
# #############################################################################


class _InMemoryCreditLedgerStore(CreditLedgerStore):
    """
    In-memory implementation of `CreditLedgerStore` using two dicts.
    """

    def __init__(self) -> None:
        self._balances: Dict[str, float] = {}
        self._debits: Dict[int, float] = {}

    def get_balance(self, account_id: str) -> Optional[float]:
        return self._balances.get(account_id)

    def set_balance(self, account_id: str, balance: float) -> None:
        self._balances[account_id] = balance

    def has_debit(self, contract_id: int) -> bool:
        return contract_id in self._debits

    def record_debit(
        self, contract_id: int, account_id: str, amount: float
    ) -> None:
        self._debits[contract_id] = amount


# #############################################################################
# CreditLedger
# #############################################################################


class CreditLedger:
    """
    Track one task-credit balance per account.
    """

    def __init__(
        self,
        *,
        free_credit: float = DEFAULT_FREE_CREDIT,
        store: Optional[CreditLedgerStore] = None,
    ) -> None:
        """
        :param free_credit: balance granted to an account on first use
        :param store: storage backend
            - Default: `_InMemoryCreditLedgerStore()`
        """
        hdbg.dassert_lte(0.0, free_credit, "free_credit must be non-negative")
        self._free_credit = free_credit
        if store is None:
            store = _InMemoryCreditLedgerStore()
        self._store = store

    def get_balance(self, account_id: str) -> float:
        """
        :return: balance of `account_id`, granting the free credit on first use
        """
        hdbg.dassert_ne(account_id, "", "account_id must be non-empty")
        balance = self._store.get_balance(account_id)
        if balance is None:
            balance = self._free_credit
            self._store.set_balance(account_id, balance)
            _LOG.info("Granted %s to new account '%s'", balance, account_id)
        return balance

    def can_afford(self, account_id: str, amount: float) -> bool:
        """
        :return: whether the balance of `account_id` covers `amount`
        """
        _LOG.debug(hprint.to_str("account_id amount"))
        hdbg.dassert_lte(0.0, amount, "amount must be non-negative")
        balance = self.get_balance(account_id)
        can_afford = balance >= amount
        _LOG.debug("return=%s", can_afford)
        return can_afford

    def debit(self, account_id: str, contract_id: int, amount: float) -> bool:
        """
        Charge `amount` to `account_id` for `contract_id`, once.

        :return: whether the balance was changed; `False` when `contract_id`
            was already debited
        """
        _LOG.debug(hprint.to_str("account_id contract_id amount"))
        hdbg.dassert_lte(0.0, amount, "amount must be non-negative")
        debited = False
        if not self._store.has_debit(contract_id):
            balance = self.get_balance(account_id) - amount
            self._store.set_balance(account_id, balance)
            self._store.record_debit(contract_id, account_id, amount)
            debited = True
        _LOG.debug("return=%s", debited)
        return debited


# #############################################################################
# Costs
# #############################################################################


def bid_cost(bid: rnbacaau.Bid) -> float:
    """
    :return: the most `bid` can cost, `n_tasks * p_max`
    """
    cost = bid.n_tasks * bid.p_max
    return cost


def fill_cost(n_tasks: int, price: float) -> float:
    """
    :return: what a cleared fill costs, `n_tasks * price`
    """
    cost = n_tasks * price
    return cost

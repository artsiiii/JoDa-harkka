import scrapy


class AutotoriMalliMerkkiScraperSpider(scrapy.Spider):
    name = 'autotori_malli_merkki_scraper'
    allowed_domains = ['auto.tori.fi']
    start_urls = ['http://auto.tori.fi/']

    def parse(self, response):
        pass
